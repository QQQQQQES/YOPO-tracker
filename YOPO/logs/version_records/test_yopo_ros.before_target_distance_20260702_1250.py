import rospy
import std_msgs.msg
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from threading import Lock
from sensor_msgs.msg import PointCloud2, PointField, Image
from sensor_msgs import point_cloud2

import cv2
import json
import os
import time
import torch
import numpy as np
import argparse
from scipy.spatial.transform import Rotation as R

from config.config import cfg
from policy.yopo_network import YopoNetwork
from policy.poly_solver import *
from policy.state_transform import *

try:
    from quadrotor_msgs.msg import PositionCommand
    POSITION_COMMAND_SOURCE = "ros:quadrotor_msgs.msg"
except ImportError:
    from control_msg import PositionCommand
    POSITION_COMMAND_SOURCE = "local:control_msg"

try:
    from traj_utils.msg import PolyTraj
    POLY_TRAJ_AVAILABLE = True
except ImportError:
    PolyTraj = None
    POLY_TRAJ_AVAILABLE = False

try:
    from torch2trt import TRTModule
except ImportError:
    print("tensorrt not found.")


class ConstantVelocityTargetFilter:
    def __init__(self, process_noise=2.0, measurement_noise=0.5):
        self.process_noise = float(process_noise)
        self.measurement_noise = float(measurement_noise)
        self.reset()

    def reset(self):
        self.x = None
        self.P = None
        self.last_time = None
        self.last_measurement_time = None

    @property
    def initialized(self):
        return self.x is not None

    def _predict_inplace(self, stamp):
        if self.x is None:
            return None
        if self.last_time is None:
            self.last_time = stamp
            return self.x[:3].copy()

        dt = float(np.clip(stamp - self.last_time, 1e-3, 1.0))
        F = np.eye(6, dtype=np.float64)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt

        q = self.process_noise
        Q = np.eye(6, dtype=np.float64) * q * dt
        Q[:3, :3] *= 0.25 * dt * dt

        self.x = F.dot(self.x)
        self.P = F.dot(self.P).dot(F.T) + Q
        self.last_time = stamp
        return self.x[:3].copy()

    def predict(self, stamp):
        return self._predict_inplace(float(stamp))

    def update(self, measurement, stamp, confidence=1.0):
        measurement = np.asarray(measurement, dtype=np.float64)
        if measurement.shape != (3,) or not np.all(np.isfinite(measurement)):
            return self.predict(stamp)

        stamp = float(stamp)
        if self.x is None:
            self.x = np.zeros(6, dtype=np.float64)
            self.x[:3] = measurement
            self.P = np.diag([0.5, 0.5, 0.5, 4.0, 4.0, 4.0]).astype(np.float64)
            self.last_time = stamp
            self.last_measurement_time = stamp
            return self.x[:3].copy()

        self._predict_inplace(stamp)
        H = np.zeros((3, 6), dtype=np.float64)
        H[:, :3] = np.eye(3, dtype=np.float64)
        conf = float(np.clip(confidence, 0.2, 1.0))
        meas_std = self.measurement_noise / conf
        Rm = np.eye(3, dtype=np.float64) * meas_std * meas_std
        innovation = measurement - H.dot(self.x)
        S = H.dot(self.P).dot(H.T) + Rm
        K = self.P.dot(H.T).dot(np.linalg.inv(S))
        self.x = self.x + K.dot(innovation)
        self.P = (np.eye(6, dtype=np.float64) - K.dot(H)).dot(self.P)
        self.last_measurement_time = stamp
        return self.x[:3].copy()

    def has_recent_measurement(self, stamp, timeout):
        if self.last_measurement_time is None:
            return False
        return (float(stamp) - self.last_measurement_time) <= float(timeout)


class YopoNet:
    def __init__(self, config, weight):
        self.config = config
        rospy.init_node('yopo_net', anonymous=False)
        rospy.loginfo(
            "Using PositionCommand from %s (md5=%s)",
            POSITION_COMMAND_SOURCE,
            getattr(PositionCommand, "_md5sum", "unknown"),
        )
        # load params
        cfg["train"] = False
        self.height = cfg['image_height']
        self.width = cfg['image_width']
        self.min_dis, self.max_dis = 0.04, 20.0
        self.goal = np.array(self.config['goal'])
        self.plan_from_reference = self.config['plan_from_reference']
        self.use_trt = self.config['use_tensorrt']
        self.verbose = self.config['verbose']
        self.visualize = self.config['visualize']
        self.image_channels = int(cfg.get('image_channels', 1))
        self.selection_mode = self.config.get('selection_mode', 'paper')
        self.objectness_threshold = float(self.config.get('objectness_threshold', 0.5))
        self.rgb_topic = self.config.get('rgb_topic', '')
        self.rgb_timeout = float(self.config.get('rgb_timeout', 0.2))
        self.target_timeout = float(self.config.get('target_timeout', 1.0))
        self.use_target_yaw = bool(self.config.get('use_target_yaw', True))
        self.use_target_prior = bool(self.config.get('use_target_prior', True))
        self.disable_arrive_check = bool(self.config.get('disable_arrive_check', False))
        self.target_filter = ConstantVelocityTargetFilter(
            process_noise=float(self.config.get('target_process_noise', 2.0)),
            measurement_noise=float(self.config.get('target_measurement_noise', 0.5)),
        )
        self.planning_min_z = self._optional_float(self.config.get('planning_min_z', None))
        self.planning_max_z = self._optional_float(self.config.get('planning_max_z', None))
        self.planning_max_abs_delta_z = self._optional_float(self.config.get('planning_max_abs_delta_z', None))
        self.publish_pos_cmd = self.config.get('publish_pos_cmd', True)
        self.publish_poly_traj = self.config.get('publish_poly_traj', False)
        self.poly_traj_topic = self.config.get('poly_traj_topic', '/planning_cmd/poly_traj')
        self.drone_id = int(self.config.get('drone_id', 0))
        self.kx = np.asarray(self.config.get('kx', [7.0, 7.0, 6.2]), dtype=np.float64)
        self.kv = np.asarray(self.config.get('kv', [4.0, 4.0, 4.0]), dtype=np.float64)
        self.Rotation_bc = R.from_euler('ZYX', [0, self.config['pitch_angle_deg'], 0], degrees=True).as_matrix()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.fx = cfg.get("fx", None)
        self.fy = cfg.get("fy", None)
        self.cx = cfg.get("cx", self.width * 0.5)
        self.cy = cfg.get("cy", self.height * 0.5)
        if self.fx is None:
            self.fx = self.width / (2.0 * np.tan(np.deg2rad(cfg["horizon_camera_fov"]) * 0.5))
        if self.fy is None:
            self.fy = self.height / (2.0 * np.tan(np.deg2rad(cfg["vertical_camera_fov"]) * 0.5))

        # variables
        self.odom = Odometry()
        self.odom_init = False
        self.rgb_image = None
        self.rgb_stamp = None
        self.last_yaw = 0.0
        self.ctrl_dt = 0.02
        self.ctrl_time = None
        self.desire_init = False
        self.arrive = False
        self.desire_pos = None
        self.desire_vel = None
        self.desire_acc = None
        self.optimal_poly_x = None
        self.optimal_poly_y = None
        self.optimal_poly_z = None
        self.lock = Lock()
        self.last_control_msg = None
        self.last_planning_clamp_debug = {}
        self.last_selection_debug = {}
        self.last_target_debug = {}
        self.poly_traj_id = 0
        self.state_transform = StateTransform()
        self.lattice_primitive = LatticePrimitive.get_instance()
        self.traj_time = self.lattice_primitive.segment_time

        # eval
        self.time_forward = 0.0
        self.time_process = 0.0
        self.time_prepare = 0.0
        self.time_interpolation = 0.0
        self.time_visualize = 0.0
        self.count = 0
        self.depth_fps = 30  # used only as processing time tolerance for printing logs

        # Load Network
        if self.use_trt:
            self.policy = TRTModule()
            self.policy.load_state_dict(torch.load(weight))
        else:
            state_dict = torch.load(weight, weights_only=True)
            self.policy = YopoNetwork()
            self.policy.load_state_dict(state_dict)
            self.policy = self.policy.to(self.device)
            self.policy.eval()
        self.warm_up()

        # ros publisher
        self.lattice_traj_pub = rospy.Publisher("/yopo_net/lattice_trajs_visual", PointCloud2, queue_size=1)
        self.best_traj_pub = rospy.Publisher("/yopo_net/best_traj_visual", PointCloud2, queue_size=1)
        self.all_trajs_pub = rospy.Publisher("/yopo_net/trajs_visual", PointCloud2, queue_size=1)
        self.tracking_debug_pub = rospy.Publisher("/yopo_net/tracking_debug", std_msgs.msg.String, queue_size=5)
        self.target_world_pub = rospy.Publisher("/yopo_net/target_world", PoseStamped, queue_size=5)
        self.ctrl_pub = rospy.Publisher(self.config["ctrl_topic"], PositionCommand, queue_size=1) if self.publish_pos_cmd else None
        self.poly_traj_pub = None
        if self.publish_poly_traj:
            if POLY_TRAJ_AVAILABLE:
                self.poly_traj_pub = rospy.Publisher(self.poly_traj_topic, PolyTraj, queue_size=1)
            else:
                rospy.logerr("publish_poly_traj is enabled, but traj_utils.msg.PolyTraj cannot be imported. "
                             "Source the MPC workspace, e.g. /home/zml/ommpc_ws/devel/setup.bash.")
        # ros subscriber
        self.odom_sub = rospy.Subscriber(self.config['odom_topic'], Odometry, self.callback_odometry, queue_size=1, tcp_nodelay=True)
        self.depth_sub = rospy.Subscriber(self.config['depth_topic'], Image, self.callback_depth, queue_size=1, tcp_nodelay=True)
        self.rgb_sub = None
        if self.rgb_topic:
            self.rgb_sub = rospy.Subscriber(self.rgb_topic, Image, self.callback_rgb, queue_size=1, tcp_nodelay=True)
        self.goal_sub = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.callback_set_goal, queue_size=1)
        # ros timer
        rospy.sleep(1.0)  # wait connection...
        self.timer_ctrl = rospy.Timer(rospy.Duration(self.ctrl_dt), self.control_pub)
        print("YOPO Net Node Ready!")
        rospy.spin()

    @staticmethod
    def _optional_float(value):
        if value is None:
            return None
        value = float(value)
        if not np.isfinite(value):
            return None
        return value

    def callback_set_goal(self, data):
        self.goal = np.asarray([data.pose.position.x, data.pose.position.y, 0.8])
        self.arrive = False
        self.desire_init = False
        self.ctrl_time = None
        self.last_control_msg = None
        self.target_filter.reset()
        print(f"New Goal: ({data.pose.position.x:.1f}, {data.pose.position.y:.1f})")

    # the first frame
    def callback_odometry(self, data):
        self.odom = data
        if not self.desire_init:
            self.desire_pos = np.array((self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z))
            self.desire_vel = np.array((self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y, self.odom.twist.twist.linear.z))
            self.desire_acc = np.array((0.0, 0.0, 0.0))
            ypr = R.from_quat([self.odom.pose.pose.orientation.x, self.odom.pose.pose.orientation.y,
                               self.odom.pose.pose.orientation.z, self.odom.pose.pose.orientation.w]).as_euler('ZYX', degrees=False)
            self.last_yaw = ypr[0]
        self.odom_init = True

        pos = np.array((self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z))
        if not self.disable_arrive_check and np.linalg.norm(pos - self.goal) < 0.5 and not self.arrive:
            print("Arrive!")
            self.arrive = True

    def callback_rgb(self, data):
        rgb = self.decode_rgb_image(data)
        with self.lock:
            self.rgb_image = rgb
            self.rgb_stamp = data.header.stamp if data.header.stamp else rospy.Time.now()

    def decode_rgb_image(self, data):
        if data.encoding in ("rgb8", "bgr8"):
            image = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, 3)
            if data.encoding == "bgr8":
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif data.encoding in ("rgba8", "bgra8"):
            image = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, 4)[:, :, :3]
            if data.encoding == "bgra8":
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif data.encoding in ("mono8", "8UC1"):
            mono = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width)
            image = np.repeat(mono[:, :, None], 3, axis=2)
        else:
            raise ValueError(f"Unsupported RGB encoding: {data.encoding}")
        if image.shape[0] != self.height or image.shape[1] != self.width:
            image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return image.astype(np.float32) / 255.0

    def get_latest_rgb(self, stamp):
        if not self.rgb_topic:
            return None
        with self.lock:
            if self.rgb_image is None or self.rgb_stamp is None:
                return None
            age = abs((stamp - self.rgb_stamp).to_sec())
            if age > self.rgb_timeout:
                return None
            return self.rgb_image.copy()

    def process_odom(self):
        # Rwb -> Rwc -> Rcw
        Rotation_wb = R.from_quat([self.odom.pose.pose.orientation.x, self.odom.pose.pose.orientation.y,
                                   self.odom.pose.pose.orientation.z, self.odom.pose.pose.orientation.w]).as_matrix()
        self.Rotation_wb = Rotation_wb
        self.Rotation_wc = np.dot(Rotation_wb, self.Rotation_bc)
        Rotation_cw = self.Rotation_wc.T

        # vel and acc
        vel_w = self.desire_vel if self.plan_from_reference else np.array([self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y, self.odom.twist.twist.linear.z])
        vel_c = np.dot(Rotation_cw, vel_w)
        acc_w = self.desire_acc
        acc_c = np.dot(Rotation_cw, acc_w)

        # goal_dir
        target_world = self.get_tracking_target_world(rospy.Time.now().to_sec(), require_recent=True, predict=False)
        goal_world = target_world if self.use_target_prior and target_world is not None else self.goal
        goal_w = goal_world - self.desire_pos
        goal_c = np.dot(Rotation_cw, goal_w)

        obs = np.concatenate((vel_c, acc_c, goal_c), axis=0).astype(np.float32)
        obs_norm = self.state_transform.normalize_obs(torch.from_numpy(obs[None, :]))
        return obs_norm

    @torch.inference_mode()
    def callback_depth(self, data):
        if not self.odom_init: return
        if self.arrive: return

        # 1. Depth Image Process (Be careful with the depth units in your application)
        time0 = time.time()
        if data.encoding == "32FC1":    # Simulator, meter
            depth = np.frombuffer(data.data, dtype=np.float32).reshape(data.height, data.width)
        elif data.encoding == "16UC1":  # RealSense, millimeter
            depth = np.frombuffer(data.data, dtype=np.uint16).reshape(data.height, data.width).astype(np.float32) / 1000.0
        else:
            raise ValueError(f"Unsupported depth encoding: {data.encoding}. Expected '32FC1' or '16UC1'.")

        if depth.shape[0] != self.height or depth.shape[1] != self.width:
            depth = cv2.resize(depth, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        depth = np.minimum(depth, self.max_dis) / self.max_dis

        # interpolated the nan value (experiment shows that treating nan directly as 0 produces similar results)
        nan_mask = np.isnan(depth) | (depth < self.min_dis / self.max_dis)
        interpolated_image = cv2.inpaint(np.uint8(depth * 255), np.uint8(nan_mask), 1, cv2.INPAINT_NS)
        interpolated_image = interpolated_image.astype(np.float32) / 255.0
        image = interpolated_image.astype(np.float32)
        stamp = data.header.stamp if data.header.stamp else rospy.Time.now()
        if self.image_channels == 1:
            depth = image.reshape([1, 1, self.height, self.width])
        elif self.image_channels == 3:
            rgb = self.get_latest_rgb(stamp)
            if rgb is None:
                depth = np.repeat(image.reshape([1, 1, self.height, self.width]), 3, axis=1)
            else:
                depth = np.transpose(rgb, (2, 0, 1)).reshape([1, 3, self.height, self.width])
        elif self.image_channels == 4:
            rgb = self.get_latest_rgb(stamp)
            if rgb is None:
                gray_rgb = np.repeat(image.reshape([1, 1, self.height, self.width]), 3, axis=1)
            else:
                gray_rgb = np.transpose(rgb, (2, 0, 1)).reshape([1, 3, self.height, self.width])
            depth = np.concatenate((gray_rgb, image.reshape([1, 1, self.height, self.width])), axis=1)
        else:
            raise ValueError(f"Unsupported image_channels={self.image_channels}. Expected 1, 3, or 4.")
        # cv2.imshow("1", depth[0][0])
        # cv2.waitKey(1)

        # 2. YOPO Network Inference
        # input prepare
        time1 = time.time()
        depth_input = torch.from_numpy(depth).to(self.device, non_blocking=True)  # (non_blocking: copying speed 3x)
        obs_norm = self.process_odom().to(self.device, non_blocking=True)
        obs_input = self.state_transform.prepare_input(obs_norm)
        # torch.cuda.synchronize()

        time2 = time.time()
        # Forward (TensorRT: inference speed increased by 5x)
        output = self.policy(depth_input, obs_input)
        target_uvd_pred = None
        if len(output) == 2:
            endstate_pred, score_pred = output
            objectness_pred = None
        else:
            endstate_pred, score_pred, objectness_logit, target_raw = output
            objectness_pred = torch.sigmoid(objectness_logit).cpu().numpy()
            target_uvd_pred = self.decode_target_prediction(target_raw.cpu().numpy())
        endstate_pred, score_pred = endstate_pred.cpu().numpy(), score_pred.cpu().numpy()
        time3 = time.time()

        # 3. Post-Processing
        # Replacing PyTorch operation on CUDA with NumPy operation on CPU (speed increased by 10x)
        endstate, score, selected_action_id = self.process_output(
            endstate_pred,
            score_pred,
            objectness_pred=objectness_pred,
            return_all_preds=self.visualize,
        )
        # Vectorization: transform the prediction(P V A in body frame) to the world frame with the attitude (without the position)
        endstate_c = endstate.reshape(-1, 3, 3).transpose(0, 2, 1)  # [N, 9] -> [N, 3, 3] -> [px vx ax, py vy ay, pz vz az]
        endstate_w = np.matmul(self.Rotation_wc, endstate_c)

        action_id = selected_action_id if self.visualize else 0
        with self.lock:  # Python3.8: threads are scheduled using time slices, add the lock to ensure safety
            odom_pos = np.array((self.odom.pose.pose.position.x,
                                 self.odom.pose.pose.position.y,
                                 self.odom.pose.pose.position.z), dtype=np.float64)
            stamp_sec = data.header.stamp.to_sec() if data.header.stamp else rospy.Time.now().to_sec()
            self.last_target_debug = self.update_tracking_target(objectness_pred, target_uvd_pred, odom_pos, stamp_sec)
            start_pos = self.desire_pos if self.plan_from_reference else np.array((self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z))
            start_vel = self.desire_vel if self.plan_from_reference else np.array((self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y, self.odom.twist.twist.linear.z))
            end_pos_w = np.array([
                endstate_w[action_id, 0, 0] + start_pos[0],
                endstate_w[action_id, 1, 0] + start_pos[1],
                endstate_w[action_id, 2, 0] + start_pos[2],
            ], dtype=np.float64)
            end_vel_w = np.array([endstate_w[action_id, 0, 1], endstate_w[action_id, 1, 1], endstate_w[action_id, 2, 1]], dtype=np.float64)
            end_acc_w = np.array([endstate_w[action_id, 0, 2], endstate_w[action_id, 1, 2], endstate_w[action_id, 2, 2]], dtype=np.float64)
            end_pos_w, end_vel_w, end_acc_w, clamp_debug = self.apply_planning_height_limits(end_pos_w, end_vel_w, end_acc_w, start_pos)
            self.last_planning_clamp_debug = clamp_debug
            self.optimal_poly_x = Poly5Solver(start_pos[0], start_vel[0], self.desire_acc[0], end_pos_w[0],
                                              end_vel_w[0], end_acc_w[0], self.traj_time)
            self.optimal_poly_y = Poly5Solver(start_pos[1], start_vel[1], self.desire_acc[1], end_pos_w[1],
                                              end_vel_w[1], end_acc_w[1], self.traj_time)
            self.optimal_poly_z = Poly5Solver(start_pos[2], start_vel[2], self.desire_acc[2], end_pos_w[2],
                                              end_vel_w[2], end_acc_w[2], self.traj_time)
            self.ctrl_time = 0.0
            self.publish_poly_traj_msg()
        self.publish_tracking_debug(score_pred, objectness_pred, selected_action_id, target_uvd_pred)
        time4 = time.time()
        self.visualize_trajectory(score_pred, endstate_w)
        time5 = time.time()

        self.print_time(time0, time1, time2, time3, time4, time5)

    def decode_target_prediction(self, target_raw):
        target_raw = np.asarray(target_raw, dtype=np.float32).reshape(3, self.lattice_primitive.traj_num).T
        return self.state_transform.decode_target_cpu(target_raw)

    def target_uvd_to_network_frame(self, target_uvd):
        target_uvd = np.asarray(target_uvd, dtype=np.float64)
        depth = float(target_uvd[2])
        if not np.isfinite(depth) or depth <= self.min_dis or depth > self.max_dis:
            return None
        x = depth
        y = -((float(target_uvd[0]) - float(self.cx)) / float(self.fx)) * depth
        z = -((float(target_uvd[1]) - float(self.cy)) / float(self.fy)) * depth
        target_vec = np.array([x, y, z], dtype=np.float64)
        if not np.all(np.isfinite(target_vec)):
            return None
        return target_vec

    def pick_target_measurement(self, objectness_pred, target_uvd_pred, odom_pos):
        if objectness_pred is None or target_uvd_pred is None:
            return None, -1, 0.0, None
        objectness = objectness_pred.reshape(self.lattice_primitive.traj_num)
        target_uvd = target_uvd_pred.reshape(self.lattice_primitive.traj_num, 3)
        detection_id = int(np.argmax(objectness))
        confidence = float(objectness[detection_id])
        if confidence < self.objectness_threshold:
            return None, detection_id, confidence, target_uvd[detection_id]
        target_vec = self.target_uvd_to_network_frame(target_uvd[detection_id])
        if target_vec is None:
            return None, detection_id, confidence, target_uvd[detection_id]
        target_world = odom_pos + self.Rotation_wc.dot(target_vec)
        return target_world, detection_id, confidence, target_uvd[detection_id]

    def update_tracking_target(self, objectness_pred, target_uvd_pred, odom_pos, stamp_sec):
        measurement, detection_id, confidence, detection_uvd = self.pick_target_measurement(
            objectness_pred,
            target_uvd_pred,
            odom_pos,
        )
        detected = measurement is not None
        if detected:
            estimate = self.target_filter.update(measurement, stamp_sec, confidence)
        else:
            estimate = self.target_filter.predict(stamp_sec)

        recent = self.target_filter.initialized and self.target_filter.has_recent_measurement(stamp_sec, self.target_timeout)
        self.publish_target_world(estimate, stamp_sec, detected, recent)
        debug = {
            "detected": bool(detected),
            "recent": bool(recent),
            "confidence": float(confidence),
            "detection_action": int(detection_id),
            "timeout": float(self.target_timeout),
        }
        if detection_uvd is not None:
            debug["detection_uvd"] = [float(x) for x in detection_uvd]
        if measurement is not None:
            debug["measurement_world"] = [float(x) for x in measurement]
        if estimate is not None:
            debug["estimate_world"] = [float(x) for x in estimate]
        return debug

    def get_tracking_target_world(self, stamp_sec, require_recent=True, predict=True):
        if not self.target_filter.initialized:
            return None
        if require_recent and not self.target_filter.has_recent_measurement(stamp_sec, self.target_timeout):
            return None
        if predict:
            return self.target_filter.predict(stamp_sec)
        return self.target_filter.x[:3].copy()

    def publish_target_world(self, target_world, stamp_sec, detected, recent):
        if target_world is None or self.target_world_pub.get_num_connections() == 0:
            return
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.from_sec(float(stamp_sec))
        msg.header.frame_id = "world"
        msg.pose.position.x = float(target_world[0])
        msg.pose.position.y = float(target_world[1])
        msg.pose.position.z = float(target_world[2])
        msg.pose.orientation.w = 1.0
        self.target_world_pub.publish(msg)

    def apply_planning_height_limits(self, end_pos_w, end_vel_w, end_acc_w, start_pos):
        raw_z = float(end_pos_w[2])
        min_z = self.planning_min_z
        max_z = self.planning_max_z
        if self.planning_max_abs_delta_z is not None and self.planning_max_abs_delta_z >= 0.0:
            delta = float(self.planning_max_abs_delta_z)
            rel_min = float(start_pos[2] - delta)
            rel_max = float(start_pos[2] + delta)
            min_z = rel_min if min_z is None else max(float(min_z), rel_min)
            max_z = rel_max if max_z is None else min(float(max_z), rel_max)
        if min_z is None and max_z is None:
            return end_pos_w, end_vel_w, end_acc_w, {
                "enabled": False,
                "clamped": False,
                "raw_terminal_z": raw_z,
                "terminal_z": raw_z,
            }
        clipped_z = raw_z
        if min_z is not None:
            clipped_z = max(clipped_z, float(min_z))
        if max_z is not None:
            clipped_z = min(clipped_z, float(max_z))
        clamped = abs(clipped_z - raw_z) > 1e-6
        if clamped:
            end_pos_w = end_pos_w.copy()
            end_vel_w = end_vel_w.copy()
            end_acc_w = end_acc_w.copy()
            end_pos_w[2] = clipped_z
            end_vel_w[2] = 0.0
            end_acc_w[2] = 0.0
        return end_pos_w, end_vel_w, end_acc_w, {
            "enabled": True,
            "clamped": bool(clamped),
            "raw_terminal_z": raw_z,
            "terminal_z": float(clipped_z),
            "min_z": None if min_z is None else float(min_z),
            "max_z": None if max_z is None else float(max_z),
        }

    def control_pub(self, _timer):
        if self.arrive:
            if self.ctrl_pub is None:
                return

            with self.lock:
                hold_pos = self.desire_pos
                if hold_pos is None:
                    hold_pos = np.array((self.odom.pose.pose.position.x,
                                         self.odom.pose.pose.position.y,
                                         self.odom.pose.pose.position.z))
                control_msg = PositionCommand()
                control_msg.header.stamp = rospy.Time.now()
                control_msg.header.frame_id = "world"
                control_msg.trajectory_flag = control_msg.TRAJECTORY_STATUS_EMPTY
                control_msg.trajectory_id = 0
                control_msg.position.x = float(hold_pos[0])
                control_msg.position.y = float(hold_pos[1])
                control_msg.position.z = float(hold_pos[2])
                control_msg.velocity.x = 0.0
                control_msg.velocity.y = 0.0
                control_msg.velocity.z = 0.0
                control_msg.acceleration.x = 0.0
                control_msg.acceleration.y = 0.0
                control_msg.acceleration.z = 0.0
                control_msg.yaw = self.last_yaw
                control_msg.yaw_dot = 0.0
                control_msg.kx = self.kx.tolist()
                control_msg.kv = self.kv.tolist()

                if hasattr(control_msg, "jerk"):
                    control_msg.jerk.x = 0.0
                    control_msg.jerk.y = 0.0
                    control_msg.jerk.z = 0.0
                if hasattr(control_msg, "angular_velocity"):
                    control_msg.angular_velocity.x = 0.0
                    control_msg.angular_velocity.y = 0.0
                    control_msg.angular_velocity.z = 0.0
                if hasattr(control_msg, "attitude"):
                    control_msg.attitude.x = 0.0
                    control_msg.attitude.y = 0.0
                    control_msg.attitude.z = self.last_yaw
                if hasattr(control_msg, "thrust"):
                    control_msg.thrust.x = 0.0
                    control_msg.thrust.y = 0.0
                    control_msg.thrust.z = 0.0
                if hasattr(control_msg, "vel_norm"):
                    control_msg.vel_norm = 0.0
                if hasattr(control_msg, "acc_norm"):
                    control_msg.acc_norm = 0.0

                self.desire_pos = np.array(hold_pos, dtype=np.float64)
                self.desire_vel = np.zeros(3)
                self.desire_acc = np.zeros(3)
                self.desire_init = True
                self.ctrl_time = None
                self.last_control_msg = control_msg
                self.ctrl_pub.publish(control_msg)
            return
        if self.ctrl_time is None or self.ctrl_time > self.traj_time:
            return

        with self.lock:  # Python3.8: threads are scheduled using time slices, add the lock to ensure safety and publish frequency
            self.ctrl_time += self.ctrl_dt
            control_msg = PositionCommand()
            control_msg.header.stamp = rospy.Time.now()
            control_msg.header.frame_id = "world"
            control_msg.trajectory_flag = control_msg.TRAJECTORY_STATUS_READY
            control_msg.trajectory_id = 0
            control_msg.position.x = self.optimal_poly_x.get_position(self.ctrl_time)
            control_msg.position.y = self.optimal_poly_y.get_position(self.ctrl_time)
            control_msg.position.z = self.optimal_poly_z.get_position(self.ctrl_time)
            control_msg.velocity.x = self.optimal_poly_x.get_velocity(self.ctrl_time)
            control_msg.velocity.y = self.optimal_poly_y.get_velocity(self.ctrl_time)
            control_msg.velocity.z = self.optimal_poly_z.get_velocity(self.ctrl_time)
            control_msg.acceleration.x = self.optimal_poly_x.get_acceleration(self.ctrl_time)
            control_msg.acceleration.y = self.optimal_poly_y.get_acceleration(self.ctrl_time)
            control_msg.acceleration.z = self.optimal_poly_z.get_acceleration(self.ctrl_time)
            self.desire_pos = np.array([control_msg.position.x, control_msg.position.y, control_msg.position.z])
            self.desire_vel = np.array([control_msg.velocity.x, control_msg.velocity.y, control_msg.velocity.z])
            self.desire_acc = np.array([control_msg.acceleration.x, control_msg.acceleration.y, control_msg.acceleration.z])
            target_world = self.get_tracking_target_world(rospy.Time.now().to_sec(), require_recent=True, predict=True)
            yaw_goal = target_world if self.use_target_yaw and target_world is not None else self.goal
            goal_dir = yaw_goal - self.desire_pos
            yaw, yaw_dot = calculate_yaw(self.desire_vel, goal_dir, self.last_yaw, self.ctrl_dt)
            self.last_yaw = yaw
            control_msg.yaw = yaw
            control_msg.yaw_dot = yaw_dot
            control_msg.kx = self.kx.tolist()
            control_msg.kv = self.kv.tolist()

            # Some controllers use an extended PositionCommand ABI.
            if hasattr(control_msg, "jerk"):
                control_msg.jerk.x = 0.0
                control_msg.jerk.y = 0.0
                control_msg.jerk.z = 0.0
            if hasattr(control_msg, "angular_velocity"):
                control_msg.angular_velocity.x = 0.0
                control_msg.angular_velocity.y = 0.0
                control_msg.angular_velocity.z = 0.0
            if hasattr(control_msg, "attitude"):
                control_msg.attitude.x = 0.0
                control_msg.attitude.y = 0.0
                control_msg.attitude.z = yaw
            if hasattr(control_msg, "thrust"):
                control_msg.thrust.x = 0.0
                control_msg.thrust.y = 0.0
                control_msg.thrust.z = 0.0
            if hasattr(control_msg, "vel_norm"):
                control_msg.vel_norm = float(np.linalg.norm(self.desire_vel))
            if hasattr(control_msg, "acc_norm"):
                control_msg.acc_norm = float(np.linalg.norm(self.desire_acc))
            self.desire_init = True
            self.last_control_msg = control_msg
            if self.ctrl_pub is not None:
                self.ctrl_pub.publish(control_msg)

    def publish_poly_traj_msg(self):
        if self.poly_traj_pub is None:
            return

        self.poly_traj_id += 1
        poly_msg = PolyTraj()
        poly_msg.drone_id = self.drone_id
        poly_msg.traj_id = self.poly_traj_id
        poly_msg.start_time = rospy.Time.now()
        poly_msg.order = 5
        poly_msg.duration = [float(self.traj_time)]
        poly_msg.coef_x = self.mpc_coef_order(self.optimal_poly_x)
        poly_msg.coef_y = self.mpc_coef_order(self.optimal_poly_y)
        poly_msg.coef_z = self.mpc_coef_order(self.optimal_poly_z)
        self.poly_traj_pub.publish(poly_msg)

    @staticmethod
    def mpc_coef_order(poly):
        return [float(poly.A[i]) for i in range(5, -1, -1)]

    def process_output(self, endstate_pred, score_pred, objectness_pred=None, return_all_preds=False):
        endstate_pred = endstate_pred.reshape(9, self.lattice_primitive.traj_num).T
        score_pred = score_pred.reshape(self.lattice_primitive.traj_num)
        objectness = None
        if objectness_pred is not None:
            objectness = objectness_pred.reshape(self.lattice_primitive.traj_num)

        if not return_all_preds:
            action_id = self.select_action(score_pred, objectness)
            lattice_id = self.lattice_primitive.traj_num - 1 - action_id
            endstate = self.state_transform.pred_to_endstate_cpu(endstate_pred[action_id, :][np.newaxis, :], lattice_id)
            score = score_pred[action_id]
            selected_action_id = action_id
        else:
            score = score_pred
            endstate = self.state_transform.pred_to_endstate_cpu(endstate_pred, torch.arange(self.lattice_primitive.traj_num-1, -1, -1))
            selected_action_id = self.select_action(score_pred, objectness)

        return endstate, score, selected_action_id

    def select_action(self, score_pred, objectness):
        self.last_selection_debug = {
            "selection_mode": self.selection_mode,
            "fallback": False,
        }
        if objectness is None or self.selection_mode == 'score':
            self.last_selection_debug["reason"] = "score_only"
            return int(np.argmin(score_pred))
        if self.selection_mode == 'objectness':
            self.last_selection_debug["reason"] = "max_objectness"
            return int(np.argmax(objectness))
        if self.selection_mode == 'hybrid':
            best_object_id = int(np.argmax(objectness))
            if float(objectness[best_object_id]) >= self.objectness_threshold:
                self.last_selection_debug["reason"] = "hybrid_objectness"
                return best_object_id
            self.last_selection_debug["fallback"] = True
            self.last_selection_debug["reason"] = "hybrid_score_fallback"
            return int(np.argmin(score_pred))
        if self.selection_mode == 'paper':
            valid = objectness >= self.objectness_threshold
            if np.any(valid):
                valid_ids = np.flatnonzero(valid)
                action_id = int(valid_ids[np.argmin(score_pred[valid_ids])])
                self.last_selection_debug["reason"] = "objectness_filter_min_score"
                self.last_selection_debug["valid_objectness_count"] = int(valid_ids.size)
                return action_id
            self.last_selection_debug["fallback"] = True
            self.last_selection_debug["reason"] = "paper_score_fallback"
            self.last_selection_debug["valid_objectness_count"] = 0
            return int(np.argmin(score_pred))
        raise ValueError(f"Unsupported selection_mode={self.selection_mode}. Expected score, objectness, hybrid, or paper.")

    def publish_tracking_debug(self, score_pred, objectness_pred, selected_action_id, target_uvd_pred=None):
        if self.tracking_debug_pub.get_num_connections() == 0:
            return
        score = score_pred.reshape(self.lattice_primitive.traj_num)
        debug = {
            "stamp": rospy.Time.now().to_sec(),
            "selection_mode": self.selection_mode,
            "selected_action": int(selected_action_id),
            "min_score": float(np.min(score)),
            "selected_score": float(score[selected_action_id]),
        }
        if self.last_selection_debug:
            debug["selection"] = self.last_selection_debug
        if objectness_pred is not None:
            obj = objectness_pred.reshape(self.lattice_primitive.traj_num)
            debug["max_objectness"] = float(np.max(obj))
            debug["selected_objectness"] = float(obj[selected_action_id])
        if target_uvd_pred is not None:
            target_uvd = target_uvd_pred.reshape(self.lattice_primitive.traj_num, 3)
            debug["selected_target_uvd"] = [float(x) for x in target_uvd[selected_action_id]]
        if self.last_target_debug:
            debug["target"] = self.last_target_debug
        if self.last_planning_clamp_debug:
            debug["planning_height"] = self.last_planning_clamp_debug
        self.tracking_debug_pub.publish(json.dumps(debug, separators=(",", ":")))

    def visualize_trajectory(self, pred_score, pred_endstate):
        dt = self.traj_time / 20.0
        start_pos = self.desire_pos if self.plan_from_reference else np.array((self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z))
        start_vel = self.desire_vel if self.plan_from_reference else np.array((self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y, self.odom.twist.twist.linear.z))
        # best predicted trajectory
        if self.best_traj_pub.get_num_connections() > 0:
            t_values = np.arange(0, self.traj_time, dt)
            points_array = np.stack((
                self.optimal_poly_x.get_position(t_values),
                self.optimal_poly_y.get_position(t_values),
                self.optimal_poly_z.get_position(t_values)
            ), axis=-1)
            header = std_msgs.msg.Header()
            header.stamp = rospy.Time.now()
            header.frame_id = 'world'
            point_cloud_msg = point_cloud2.create_cloud_xyz32(header, points_array)
            self.best_traj_pub.publish(point_cloud_msg)
        # lattice primitive
        if self.visualize and self.lattice_traj_pub.get_num_connections() > 0:
            lattice_endstate = self.lattice_primitive.lattice_pos_node.cpu().numpy()
            lattice_endstate = np.dot(lattice_endstate, self.Rotation_wc.T)
            zero_state = np.zeros_like(lattice_endstate)
            lattice_poly_x = Polys5Solver(start_pos[0], start_vel[0], self.desire_acc[0],
                                          lattice_endstate[:, 0] + start_pos[0], zero_state[:, 0], zero_state[:, 0], self.traj_time)
            lattice_poly_y = Polys5Solver(start_pos[1], start_vel[1], self.desire_acc[1],
                                          lattice_endstate[:, 1] + start_pos[1], zero_state[:, 1], zero_state[:, 1], self.traj_time)
            lattice_poly_z = Polys5Solver(start_pos[2], start_vel[2], self.desire_acc[2],
                                          lattice_endstate[:, 2] + start_pos[2], zero_state[:, 2], zero_state[:, 2], self.traj_time)
            t_values = np.arange(0, self.traj_time, dt)
            points_array = np.stack((
                lattice_poly_x.get_position(t_values),
                lattice_poly_y.get_position(t_values),
                lattice_poly_z.get_position(t_values)
            ), axis=-1)
            header = std_msgs.msg.Header()
            header.stamp = rospy.Time.now()
            header.frame_id = 'world'
            point_cloud_msg = point_cloud2.create_cloud_xyz32(header, points_array)
            self.lattice_traj_pub.publish(point_cloud_msg)
        # all predicted trajectories
        if self.visualize and self.all_trajs_pub.get_num_connections() > 0:
            all_poly_x = Polys5Solver(start_pos[0], start_vel[0], self.desire_acc[0],
                                      pred_endstate[:, 0, 0] + start_pos[0], pred_endstate[:, 0, 1], pred_endstate[:, 0, 2], self.traj_time)
            all_poly_y = Polys5Solver(start_pos[1], start_vel[1], self.desire_acc[1],
                                      pred_endstate[:, 1, 0] + start_pos[1], pred_endstate[:, 1, 1], pred_endstate[:, 1, 2], self.traj_time)
            all_poly_z = Polys5Solver(start_pos[2], start_vel[2], self.desire_acc[2],
                                      pred_endstate[:, 2, 0] + start_pos[2], pred_endstate[:, 2, 1], pred_endstate[:, 2, 2], self.traj_time)
            t_values = np.arange(0, self.traj_time, dt)
            points_array = np.stack((
                all_poly_x.get_position(t_values),
                all_poly_y.get_position(t_values),
                all_poly_z.get_position(t_values)
            ), axis=-1)
            scores = np.repeat(pred_score, t_values.size)
            points_array = np.column_stack((points_array, scores))
            header = std_msgs.msg.Header()
            header.stamp = rospy.Time.now()
            header.frame_id = 'world'
            fields = [PointField('x', 0, PointField.FLOAT32, 1), PointField('y', 4, PointField.FLOAT32, 1),
                      PointField('z', 8, PointField.FLOAT32, 1), PointField('intensity', 12, PointField.FLOAT32, 1)]
            point_cloud_msg = point_cloud2.create_cloud(header, fields, points_array)
            self.all_trajs_pub.publish(point_cloud_msg)

    def print_time(self, time0, time1, time2, time3, time4, time5):
        """
        Performance reference: PyTorch model should take < 5 ms; TensorRT model should take < 1 ms

        Notes:
        - Running program and enabling RViz under WSL greatly increase processing time, and Ubuntu does not have these issues
        - Even with queue_size=1, it may cause message accumulation and lag when processing time exceeds the image frequency
        """
        self.time_interpolation = self.time_interpolation + (time1 - time0)
        self.time_prepare = self.time_prepare + (time2 - time1)
        self.time_forward = self.time_forward + (time3 - time2)
        self.time_process = self.time_process + (time4 - time3)
        self.time_visualize = self.time_visualize + (time5 - time4)
        self.count = self.count + 1

        total_time = (time5 - time0) * 1000
        tolerance = 1000.0 / self.depth_fps
        if total_time > tolerance:
            rospy.logwarn(f"Warn: Processing time {(time5 - time0) * 1000:.2f} ms exceeds {tolerance:.2f} ms, may cause message lag!")
            print(f"\033[34mCurrent Time Consuming:\033[0m "
                  f"depth-interpolation: \033[32m{1000 * (time1 - time0):.2f} ms\033[0m; "
                  f"data-prepare: \033[32m{1000 * (time2 - time1):.2f} ms\033[0m; "
                  f"network-inference: \033[32m{1000 * (time3 - time2):.2f} ms\033[0m; "
                  f"post-process: \033[32m{1000 * (time4 - time3):.2f} ms\033[0m; "
                  f"visualize-trajectory: \033[32m{1000 * (time5 - time4):.2f} ms\033[0m")
        if self.verbose or (total_time > tolerance):
            print(f"\033[34mAverage Time Consuming:\033[0m "
                  f"depth-interpolation: \033[32m{1000 * self.time_interpolation / self.count:.2f} ms\033[0m; "
                  f"data-prepare: \033[32m{1000 * self.time_prepare / self.count:.2f} ms\033[0m; "
                  f"network-inference: \033[32m{1000 * self.time_forward / self.count:.2f} ms\033[0m; "
                  f"post-process: \033[32m{1000 * self.time_process / self.count:.2f} ms\033[0m; "
                  f"visualize-trajectory: \033[32m{1000 * self.time_visualize / self.count:.2f} ms\033[0m")

    def warm_up(self):
        depth = torch.zeros((1, self.image_channels, self.height, self.width), dtype=torch.float32, device=self.device)
        obs = torch.zeros((1, 9), dtype=torch.float32, device=self.device)
        obs = self.state_transform.prepare_input(obs)
        output = self.policy(depth, obs)
        endstate_pred = output[0]
        _ = self.state_transform.pred_to_endstate(endstate_pred)


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_tensorrt", type=int, default=0, help="use tensorrt or not")
    parser.add_argument("--trial", type=int, default=1, help="trial number")
    parser.add_argument("--epoch", type=int, default=50, help="epoch number")
    parser.add_argument("--weight", type=str, default="", help="explicit checkpoint path")
    parser.add_argument("--goal", type=float, nargs=3, default=[0.0, 0.0, 2.0], help="world-frame goal/target prior")
    parser.add_argument("--odom_topic", type=str, default="/LIVO2/imu_propagate")
    parser.add_argument("--depth_topic", type=str, default="/iris_0/realsense/depth_camera/depth/image_raw")
    parser.add_argument("--rgb_topic", type=str, default="")
    parser.add_argument("--rgb_timeout", type=float, default=0.2)
    parser.add_argument("--ctrl_topic", type=str, default="/planning/pos_cmd")
    parser.add_argument("--plan_from_reference", type=int, default=0)
    parser.add_argument("--publish_pos_cmd", type=int, default=1)
    parser.add_argument("--publish_poly_traj", type=int, default=0)
    parser.add_argument("--poly_traj_topic", type=str, default="/planning_cmd/poly_traj")
    parser.add_argument("--pitch_angle_deg", type=float, default=0.0)
    parser.add_argument("--selection_mode", type=str, default="paper", choices=["score", "objectness", "hybrid", "paper"])
    parser.add_argument("--objectness_threshold", type=float, default=0.5)
    parser.add_argument("--target_timeout", type=float, default=1.0)
    parser.add_argument("--target_process_noise", type=float, default=2.0)
    parser.add_argument("--target_measurement_noise", type=float, default=0.5)
    parser.add_argument("--use_target_yaw", type=int, default=1)
    parser.add_argument("--use_target_prior", type=int, default=1)
    parser.add_argument("--disable_arrive_check", type=int, default=0)
    parser.add_argument("--planning_min_z", type=float, default=float("nan"), help="optional absolute minimum terminal z")
    parser.add_argument("--planning_max_z", type=float, default=float("nan"), help="optional absolute maximum terminal z")
    parser.add_argument("--planning_max_abs_delta_z", type=float, default=float("nan"), help="optional max terminal z change relative to current/start z")
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--visualize", type=int, default=1)
    return parser


if __name__ == "__main__":
    args = parser().parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if args.weight:
        weight = args.weight
    else:
        weight = "yopo_trt.pth" if args.use_tensorrt else base_dir + "/saved/YOPO_{}/epoch{}.pth".format(args.trial, args.epoch)
    print("load weight from:", weight)

    settings = {'use_tensorrt': args.use_tensorrt,
                'goal': args.goal,      # 目标点位置
                'pitch_angle_deg': args.pitch_angle_deg,   # 相机俯仰角(仰为负)
                'odom_topic': args.odom_topic,                   # 里程计话题
                'depth_topic': args.depth_topic,               # 深度图话题
                'rgb_topic': args.rgb_topic,
                'rgb_timeout': args.rgb_timeout,
                'ctrl_topic': args.ctrl_topic,        # 控制器话题
                'publish_pos_cmd': bool(args.publish_pos_cmd),      # 使用MPC控制时关闭PositionCommand输出
                'publish_poly_traj': bool(args.publish_poly_traj),     # 输出traj_utils/PolyTraj给MPC
                'poly_traj_topic': args.poly_traj_topic,
                'drone_id': 0,
                'plan_from_reference': bool(args.plan_from_reference),   # 从参考状态规划？位置控制器: True, 神经网络直接控制: False
                'selection_mode': args.selection_mode,
                'objectness_threshold': args.objectness_threshold,
                'target_timeout': args.target_timeout,
                'target_process_noise': args.target_process_noise,
                'target_measurement_noise': args.target_measurement_noise,
                'use_target_yaw': bool(args.use_target_yaw),
                'use_target_prior': bool(args.use_target_prior),
                'disable_arrive_check': bool(args.disable_arrive_check),
                'planning_min_z': args.planning_min_z,
                'planning_max_z': args.planning_max_z,
                'planning_max_abs_delta_z': args.planning_max_abs_delta_z,
                'verbose': bool(args.verbose),               # 打印耗时？
                'visualize': bool(args.visualize)               # 可视化所有轨迹？(实飞改为False节省计算)
                }
    YopoNet(settings, weight)
