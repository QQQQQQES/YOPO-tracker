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
import sys
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

    def predict_position(self, stamp):
        if self.x is None:
            return None
        if self.last_time is None:
            return self.x[:3].copy()
        dt = float(np.clip(float(stamp) - self.last_time, 1e-3, 1.0))
        predicted = self.x.copy()
        predicted[0] += dt * predicted[3]
        predicted[1] += dt * predicted[4]
        predicted[2] += dt * predicted[5]
        return predicted[:3].copy()

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
        self.perf_log_interval = float(self.config.get('perf_log_interval', 1.0))
        self.visualize = self.config['visualize']
        self.image_channels = int(cfg.get('image_channels', 1))
        self.selection_mode = self.config.get('selection_mode', 'paper')
        self.objectness_threshold = float(self.config.get('objectness_threshold', 0.5))
        self.target_selection_score_margin = float(self.config.get('target_selection_score_margin', 4.5))
        self.target_consistency_gate = float(self.config.get('target_consistency_gate', 3.0))
        self.target_nms_distance = float(self.config.get('target_nms_distance', 1.0))
        self.target_timeout = float(self.config.get('target_timeout', 1.0))
        self.rgb_topic = self.config.get('rgb_topic', '')
        self.rgb_timeout = float(self.config.get('rgb_timeout', 0.2))
        self.target_truth_topic = self.config.get('target_truth_topic', '')
        self.target_truth_timeout = float(self.config.get('target_truth_timeout', 1.0))
        self.target_truth_follow_z = self._optional_float(self.config.get('target_truth_follow_z', None))
        self.use_truth_target_in_paper = bool(self.config.get('use_truth_target_in_paper', False))
        self.no_target_behavior = str(self.config.get('no_target_behavior', 'fallback')).lower()
        self.no_target_timeout = float(self.config.get('no_target_timeout', self.target_timeout))
        if self.no_target_behavior not in ('fallback', 'hold'):
            raise ValueError("Unsupported no_target_behavior={}. Expected fallback or hold.".format(self.no_target_behavior))
        self.lost_target_progress_enabled = bool(self.config.get('lost_target_progress_enabled', True))
        self.lost_target_progress_timeout = float(self.config.get('lost_target_progress_timeout', self.target_timeout))
        self.lost_target_progress_weight = float(self.config.get('lost_target_progress_weight', 0.05))
        self.lost_target_progress_cap = float(self.config.get(
            'lost_target_progress_cap',
            cfg.get('lost_forward_progress_cap', cfg.get('tracking_progress_cap', cfg.get('goal_length', 8.0))),
        ))
        self.lost_target_progress_xy_only = bool(self.config.get('lost_target_progress_xy_only', True))
        self.use_target_yaw = bool(self.config.get('use_target_yaw', True))
        self.max_yaw_rate_deg_s = float(self.config.get('max_yaw_rate_deg_s', cfg.get('max_yaw_rate_deg_s', 90.0)))
        self.max_yaw_rate_pi_s = self.max_yaw_rate_deg_s / 180.0
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
        self.target_camera_Rcl = self._optional_matrix(
            self.config.get('target_camera_Rcl', self.config.get('Rcl', None)),
            (3, 3),
            'target_camera_Rcl',
        )
        self.target_camera_Pcl = self._optional_matrix(
            self.config.get('target_camera_Pcl', self.config.get('Pcl', None)),
            (3,),
            'target_camera_Pcl',
        )
        if (self.target_camera_Rcl is None) != (self.target_camera_Pcl is None):
            raise ValueError("target_camera_Rcl and target_camera_Pcl must be configured together")
        self.use_target_camera_extrinsic = self.target_camera_Rcl is not None

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
        self.last_consistent_action_ids = np.empty(0, dtype=np.int64)
        self.last_no_target_debug = {}
        self.no_target_hold_active = False
        self.no_target_hold_pos = None
        self.target_truth_world = None
        self.target_truth_stamp = None
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
        self.last_perf_log_time = 0.0

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
        self.target_truth_sub = None
        if self.target_truth_topic:
            self.target_truth_sub = rospy.Subscriber(self.target_truth_topic, PoseStamped, self.callback_target_truth, queue_size=1, tcp_nodelay=True)
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

    @staticmethod
    def _optional_matrix(value, shape, name):
        if value is None:
            return None
        arr = np.asarray(value, dtype=np.float64)
        if arr.size != int(np.prod(shape)):
            raise ValueError("{} must have {} values, got {}".format(name, int(np.prod(shape)), arr.size))
        arr = arr.reshape(shape)
        if not np.all(np.isfinite(arr)):
            raise ValueError("{} contains non-finite values".format(name))
        return arr

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

    def callback_target_truth(self, data):
        target = np.array([
            data.pose.position.x,
            data.pose.position.y,
            data.pose.position.z,
        ], dtype=np.float64)
        if self.target_truth_follow_z is not None:
            target[2] = float(self.target_truth_follow_z)
        with self.lock:
            self.target_truth_world = target
            self.target_truth_stamp = data.header.stamp if data.header.stamp else rospy.Time.now()

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

        obs = np.concatenate((vel_c, acc_c), axis=0).astype(np.float32)
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
            return_all_preds=self.visualize or self.selection_mode in ('paper', 'target_distance'),
        )
        # Vectorization: transform the prediction(P V A in body frame) to the world frame with the attitude (without the position)
        endstate_c = endstate.reshape(-1, 3, 3).transpose(0, 2, 1)  # [N, 9] -> [N, 3, 3] -> [px vx ax, py vy ay, pz vz az]
        endstate_w = np.matmul(self.Rotation_wc, endstate_c)

        action_id = selected_action_id if self.visualize else 0
        no_target_hold = False
        with self.lock:  # Python3.8: threads are scheduled using time slices, add the lock to ensure safety
            odom_pos = np.array((self.odom.pose.pose.position.x,
                                 self.odom.pose.pose.position.y,
                                 self.odom.pose.pose.position.z), dtype=np.float64)
            stamp_sec = data.header.stamp.to_sec() if data.header.stamp else rospy.Time.now().to_sec()
            self.last_target_debug = self.update_tracking_target(objectness_pred, target_uvd_pred, odom_pos, stamp_sec)
            if self.should_hold_for_no_target(stamp_sec):
                self.activate_no_target_hold(odom_pos, stamp_sec)
                self.last_selection_debug = {
                    "selection_mode": self.selection_mode,
                    "fallback": True,
                    "reason": "no_target_hold",
                    "no_target_behavior": self.no_target_behavior,
                    "no_target_timeout": float(self.no_target_timeout),
                }
                self.last_planning_clamp_debug = {
                    "enabled": False,
                    "clamped": False,
                    "reason": "no_target_hold",
                }
                no_target_hold = True
            else:
                self.clear_no_target_hold()
            start_pos = self.desire_pos if self.plan_from_reference else np.array((self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z))
            start_vel = self.desire_vel if self.plan_from_reference else np.array((self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y, self.odom.twist.twist.linear.z))
            if (not no_target_hold) and self.selection_mode == 'paper':
                selected_action_id = self.select_action_by_paper_consistency(
                    score_pred,
                    objectness_pred,
                    endstate_w,
                    start_pos,
                    stamp_sec,
                )
                action_id = selected_action_id
            elif (not no_target_hold) and self.selection_mode == 'target_distance':
                selected_action_id = self.select_action_by_target_distance(
                    score_pred,
                    objectness_pred,
                    endstate_w,
                    start_pos,
                    stamp_sec,
                )
                action_id = selected_action_id
            if no_target_hold:
                pass
            else:
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
        if not no_target_hold:
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
        right = ((float(target_uvd[0]) - float(self.cx)) / float(self.fx)) * depth
        down = ((float(target_uvd[1]) - float(self.cy)) / float(self.fy)) * depth
        if self.use_target_camera_extrinsic:
            target_optical = np.array([right, down, depth], dtype=np.float64)
            target_vec = self.target_camera_Rcl.T.dot(target_optical - self.target_camera_Pcl)
        else:
            target_vec = np.array([depth, -right, -down], dtype=np.float64)
        if not np.all(np.isfinite(target_vec)):
            return None
        return target_vec

    def build_target_candidates(self, objectness_pred, target_uvd_pred, odom_pos):
        if objectness_pred is None or target_uvd_pred is None:
            return []
        objectness = objectness_pred.reshape(self.lattice_primitive.traj_num)
        target_uvd = target_uvd_pred.reshape(self.lattice_primitive.traj_num, 3)
        valid_objectness = np.isfinite(objectness) & (objectness >= self.objectness_threshold)
        candidates = []
        for detection_id in np.flatnonzero(valid_objectness):
            target_vec = self.target_uvd_to_network_frame(target_uvd[detection_id])
            if target_vec is None:
                continue
            target_world = odom_pos + self.Rotation_wc.dot(target_vec)
            if not np.all(np.isfinite(target_world)):
                continue
            candidates.append({
                "id": int(detection_id),
                "confidence": float(objectness[detection_id]),
                "uvd": target_uvd[detection_id].copy(),
                "world": target_world,
            })
        return candidates

    def filter_target_candidates(self, candidates, stamp_sec):
        predicted = self.target_filter.predict_position(stamp_sec)
        recent = (
            self.target_filter.initialized
            and self.target_filter.has_recent_measurement(stamp_sec, self.target_timeout)
        )
        for candidate in candidates:
            if predicted is None:
                candidate["consistency_distance"] = float("nan")
            else:
                candidate["consistency_distance"] = float(np.linalg.norm(candidate["world"] - predicted))

        consistency_candidates = list(candidates)
        gate_enabled = bool(recent and predicted is not None and self.target_consistency_gate > 0.0)
        if gate_enabled:
            consistency_candidates = [
                candidate
                for candidate in candidates
                if candidate["consistency_distance"] <= self.target_consistency_gate
            ]

        if gate_enabled:
            ordered = sorted(consistency_candidates, key=lambda c: (c["consistency_distance"], -c["confidence"]))
        else:
            ordered = sorted(consistency_candidates, key=lambda c: -c["confidence"])

        kept = []
        nms_enabled = self.target_nms_distance > 0.0
        for candidate in ordered:
            if nms_enabled and any(np.linalg.norm(candidate["world"] - other["world"]) <= self.target_nms_distance for other in kept):
                continue
            kept.append(candidate)

        debug = {
            "candidate_count": int(len(candidates)),
            "consistent_count": int(len(consistency_candidates)),
            "nms_count": int(len(kept)),
            "consistency_gate": float(self.target_consistency_gate),
            "nms_distance": float(self.target_nms_distance),
            "ekf_recent": bool(recent),
        }
        if predicted is not None:
            debug["predicted_world"] = [float(x) for x in predicted]
        debug["valid_action_ids"] = [int(candidate["id"]) for candidate in kept]
        return kept, debug

    def update_tracking_target(self, objectness_pred, target_uvd_pred, odom_pos, stamp_sec):
        raw_objectness = None
        raw_best_id = -1
        raw_best_confidence = 0.0
        raw_best_uvd = None
        if objectness_pred is not None and target_uvd_pred is not None:
            raw_objectness = objectness_pred.reshape(self.lattice_primitive.traj_num)
            raw_target_uvd = target_uvd_pred.reshape(self.lattice_primitive.traj_num, 3)
            raw_best_id = int(np.argmax(raw_objectness))
            raw_best_confidence = float(raw_objectness[raw_best_id])
            raw_best_uvd = raw_target_uvd[raw_best_id]

        candidates = self.build_target_candidates(objectness_pred, target_uvd_pred, odom_pos)
        kept_candidates, candidate_debug = self.filter_target_candidates(candidates, stamp_sec)
        self.last_consistent_action_ids = np.asarray([candidate["id"] for candidate in kept_candidates], dtype=np.int64)

        selected_candidate = kept_candidates[0] if kept_candidates else None
        measurement = None if selected_candidate is None else selected_candidate["world"]
        detection_id = raw_best_id if selected_candidate is None else int(selected_candidate["id"])
        confidence = raw_best_confidence if selected_candidate is None else float(selected_candidate["confidence"])
        detection_uvd = raw_best_uvd if selected_candidate is None else selected_candidate["uvd"]
        detected = selected_candidate is not None
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
        debug.update(candidate_debug)
        if detection_uvd is not None:
            debug["detection_uvd"] = [float(x) for x in detection_uvd]
        if measurement is not None:
            debug["measurement_world"] = [float(x) for x in measurement]
        if estimate is not None:
            debug["estimate_world"] = [float(x) for x in estimate]
        return debug

    def get_tracking_target_world(self, stamp_sec, require_recent=True, predict=True):
        if self.allow_truth_target():
            truth_target = self.get_truth_target_world(stamp_sec, require_recent=require_recent)
            if truth_target is not None:
                return truth_target
        if not self.target_filter.initialized:
            return None
        if require_recent and not self.target_filter.has_recent_measurement(stamp_sec, self.target_timeout):
            return None
        if predict:
            return self.target_filter.predict(stamp_sec)
        return self.target_filter.x[:3].copy()

    def get_lost_target_world(self, stamp_sec):
        if not self.lost_target_progress_enabled:
            return None
        if self.allow_truth_target():
            truth_target = self.get_truth_target_world(stamp_sec, require_recent=True)
            if truth_target is not None:
                return truth_target
        if not self.target_filter.initialized:
            return None
        if not self.target_filter.has_recent_measurement(stamp_sec, self.lost_target_progress_timeout):
            return None
        return self.target_filter.predict_position(stamp_sec)

    def get_truth_target_world(self, stamp_sec, require_recent=True):
        if self.target_truth_world is None or self.target_truth_stamp is None:
            return None
        age = abs(float(stamp_sec) - self.target_truth_stamp.to_sec())
        if require_recent and age > self.target_truth_timeout:
            return None
        return self.target_truth_world.copy()

    def allow_truth_target(self):
        if not self.target_truth_topic:
            return False
        return self.selection_mode != 'paper' or self.use_truth_target_in_paper

    def has_recent_tracking_target(self, stamp_sec, timeout=None):
        if self.allow_truth_target():
            if self.get_truth_target_world(stamp_sec, require_recent=True) is not None:
                return True
        if not self.target_filter.initialized:
            return False
        timeout = self.target_timeout if timeout is None else float(timeout)
        return self.target_filter.has_recent_measurement(stamp_sec, timeout)

    def should_hold_for_no_target(self, stamp_sec):
        if self.no_target_behavior != 'hold':
            return False
        if self.get_lost_target_world(stamp_sec) is not None:
            return False
        return not self.has_recent_tracking_target(stamp_sec, timeout=self.no_target_timeout)

    def activate_no_target_hold(self, hold_pos, stamp_sec):
        hold_pos = np.asarray(hold_pos, dtype=np.float64)
        if not self.no_target_hold_active or self.no_target_hold_pos is None:
            self.no_target_hold_pos = hold_pos.copy()
        self.no_target_hold_active = True
        self.desire_pos = self.no_target_hold_pos.copy()
        self.desire_vel = np.zeros(3)
        self.desire_acc = np.zeros(3)
        self.desire_init = True
        self.ctrl_time = None
        self.last_no_target_debug = {
            "behavior": self.no_target_behavior,
            "active": True,
            "reason": "no_recent_target_hold",
            "timeout": float(self.no_target_timeout),
            "hold_position": [float(x) for x in self.no_target_hold_pos],
            "stamp": float(stamp_sec),
        }

    def clear_no_target_hold(self):
        if self.no_target_hold_active:
            self.last_no_target_debug = {
                "behavior": self.no_target_behavior,
                "active": False,
                "reason": "target_reacquired",
            }
        self.no_target_hold_active = False
        self.no_target_hold_pos = None

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
        min_z, max_z = self.effective_planning_height_bounds(start_pos)
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

    def effective_planning_height_bounds(self, start_pos):
        min_z = self.planning_min_z
        max_z = self.planning_max_z
        if self.planning_max_abs_delta_z is not None and self.planning_max_abs_delta_z >= 0.0:
            delta = float(self.planning_max_abs_delta_z)
            rel_min = float(start_pos[2] - delta)
            rel_max = float(start_pos[2] + delta)
            min_z = rel_min if min_z is None else max(float(min_z), rel_min)
            max_z = rel_max if max_z is None else min(float(max_z), rel_max)
        return min_z, max_z

    def clamp_candidate_terminal_z(self, end_positions, start_pos):
        min_z, max_z = self.effective_planning_height_bounds(start_pos)
        if min_z is None and max_z is None:
            return end_positions
        end_positions = end_positions.copy()
        if min_z is not None:
            end_positions[:, 2] = np.maximum(end_positions[:, 2], float(min_z))
        if max_z is not None:
            end_positions[:, 2] = np.minimum(end_positions[:, 2], float(max_z))
        return end_positions

    def control_pub(self, _timer):
        if self.arrive or self.no_target_hold_active:
            if self.ctrl_pub is None:
                return

            with self.lock:
                hold_pos = self.no_target_hold_pos if self.no_target_hold_active else self.desire_pos
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
            yaw, yaw_dot = calculate_yaw(
                self.desire_vel,
                goal_dir,
                self.last_yaw,
                self.ctrl_dt,
                max_yaw_rate=self.max_yaw_rate_pi_s,
            )
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
        if self.selection_mode == 'target_distance':
            valid = objectness >= self.objectness_threshold
            if np.any(valid):
                valid_ids = np.flatnonzero(valid)
                action_id = int(valid_ids[np.argmin(score_pred[valid_ids])])
                self.last_selection_debug["reason"] = "target_distance_preselect_paper"
                self.last_selection_debug["valid_objectness_count"] = int(valid_ids.size)
                return action_id
            self.last_selection_debug["fallback"] = True
            self.last_selection_debug["reason"] = "target_distance_preselect_score"
            self.last_selection_debug["valid_objectness_count"] = 0
            return int(np.argmin(score_pred))
        raise ValueError(f"Unsupported selection_mode={self.selection_mode}. Expected score, objectness, hybrid, paper, or target_distance.")

    def select_action_by_lost_target_progress(self, score_pred, endstate_w, start_pos, stamp_sec, reason):
        target_world = self.get_lost_target_world(stamp_sec)
        if target_world is None or endstate_w is None or endstate_w.shape[0] != self.lattice_primitive.traj_num:
            return None

        score = np.asarray(score_pred, dtype=np.float64).reshape(self.lattice_primitive.traj_num)
        finite = np.isfinite(score)
        valid_ids = np.flatnonzero(finite)
        if valid_ids.size == 0:
            return None

        start_pos = np.asarray(start_pos, dtype=np.float64)
        target_vec = np.asarray(target_world, dtype=np.float64) - start_pos
        if self.lost_target_progress_xy_only:
            target_vec = target_vec.copy()
            target_vec[2] = 0.0
        target_norm = float(np.linalg.norm(target_vec))
        if target_norm < 1e-6:
            return None
        target_dir = target_vec / target_norm

        end_positions = start_pos.reshape(1, 3) + endstate_w[:, :, 0]
        selection_end_positions = self.clamp_candidate_terminal_z(end_positions, start_pos)
        terminal_vec = selection_end_positions[valid_ids] - start_pos.reshape(1, 3)
        if self.lost_target_progress_xy_only:
            terminal_vec = terminal_vec.copy()
            terminal_vec[:, 2] = 0.0

        progress = terminal_vec.dot(target_dir)
        cap = max(float(self.lost_target_progress_cap), 1e-6)
        progress_cost = cap - np.clip(progress, 0.0, cap)
        combined_cost = score[valid_ids] + float(self.lost_target_progress_weight) * progress_cost
        best_pos = int(np.argmin(combined_cost))
        action_id = int(valid_ids[best_pos])

        selected_terminal_vec = terminal_vec[best_pos]
        selected_progress = float(progress[best_pos])
        selected_lateral_error = float(np.linalg.norm(selected_terminal_vec - selected_progress * target_dir))
        self.last_selection_debug.update({
            "fallback": True,
            "reason": reason,
            "valid_score_count": int(valid_ids.size),
            "lost_target_world": [float(x) for x in target_world],
            "lost_target_progress_weight": float(self.lost_target_progress_weight),
            "lost_target_progress_cap": float(cap),
            "selected_terminal_progress": selected_progress,
            "selected_terminal_lateral_error": selected_lateral_error,
            "selected_score": float(score[action_id]),
            "selected_progress_cost": float(progress_cost[best_pos]),
            "selected_combined_cost": float(combined_cost[best_pos]),
        })
        return action_id

    def select_action_by_paper_consistency(self, score_pred, objectness_pred, endstate_w=None, start_pos=None, stamp_sec=None):
        score = score_pred.reshape(self.lattice_primitive.traj_num)
        finite = np.isfinite(score)
        objectness = None if objectness_pred is None else objectness_pred.reshape(self.lattice_primitive.traj_num)
        valid_ids = self.last_consistent_action_ids
        valid_ids = valid_ids[(valid_ids >= 0) & (valid_ids < self.lattice_primitive.traj_num)] if valid_ids.size else valid_ids
        valid_ids = valid_ids[finite[valid_ids]] if valid_ids.size else valid_ids
        self.last_selection_debug = {
            "selection_mode": self.selection_mode,
            "fallback": False,
            "objectness_threshold": float(self.objectness_threshold),
            "valid_consistent_count": int(valid_ids.size),
        }
        if valid_ids.size:
            action_id = int(valid_ids[np.argmin(score[valid_ids])])
            self.last_selection_debug.update({
                "reason": "paper_ekf_nms_min_score",
                "valid_action_ids": [int(x) for x in valid_ids],
            })
            return action_id

        self.last_selection_debug["fallback"] = True
        if endstate_w is not None and start_pos is not None and stamp_sec is not None:
            lost_action_id = self.select_action_by_lost_target_progress(
                score,
                endstate_w,
                start_pos,
                stamp_sec,
                "paper_lost_target_ekf_progress",
            )
            if lost_action_id is not None:
                return lost_action_id
        if objectness is not None:
            raw_valid = np.flatnonzero(np.isfinite(objectness) & (objectness >= self.objectness_threshold) & finite)
            self.last_selection_debug["raw_objectness_valid_count"] = int(raw_valid.size)
            if raw_valid.size:
                action_id = int(raw_valid[np.argmin(score[raw_valid])])
                self.last_selection_debug["reason"] = "paper_objectness_min_score_fallback"
                self.last_selection_debug["valid_action_ids"] = [int(x) for x in raw_valid]
                return action_id
        self.last_selection_debug["reason"] = "paper_no_objectness_score_fallback"
        return int(np.argmin(score))

    def select_action_by_target_distance(self, score_pred, objectness_pred, endstate_w, start_pos, stamp_sec):
        score = score_pred.reshape(self.lattice_primitive.traj_num)
        objectness = None if objectness_pred is None else objectness_pred.reshape(self.lattice_primitive.traj_num)
        fallback_id = self.select_action(score, objectness)
        target_world = self.get_tracking_target_world(stamp_sec, require_recent=True)
        if target_world is None or endstate_w.shape[0] != self.lattice_primitive.traj_num:
            lost_action_id = self.select_action_by_lost_target_progress(
                score,
                endstate_w,
                start_pos,
                stamp_sec,
                "target_distance_lost_target_ekf_progress",
            )
            if lost_action_id is not None:
                return lost_action_id
            self.last_selection_debug["fallback"] = True
            self.last_selection_debug["reason"] = "target_distance_no_recent_target"
            return fallback_id

        end_positions = start_pos.reshape(1, 3) + endstate_w[:, :, 0]
        selection_end_positions = self.clamp_candidate_terminal_z(end_positions, start_pos)
        finite = np.isfinite(score)
        min_score = float(np.min(score[finite])) if np.any(finite) else float("nan")
        score_valid = finite.copy()
        if np.isfinite(min_score) and self.target_selection_score_margin >= 0.0:
            score_valid &= score <= min_score + self.target_selection_score_margin
        valid = score_valid.copy()
        objectness_valid = None
        if objectness is not None:
            objectness_valid = np.isfinite(objectness) & (objectness >= self.objectness_threshold)
            valid &= objectness_valid
        if not np.any(valid):
            self.last_selection_debug["fallback"] = True
            if not np.any(score_valid):
                self.last_selection_debug["reason"] = "target_distance_no_score_valid_candidate"
            elif objectness_valid is not None:
                self.last_selection_debug["reason"] = "target_distance_no_objectness_valid_candidate"
            else:
                self.last_selection_debug["reason"] = "target_distance_no_valid_candidate"
            self.last_selection_debug["valid_score_count"] = int(np.count_nonzero(score_valid))
            if objectness_valid is not None:
                self.last_selection_debug["valid_objectness_count"] = int(np.count_nonzero(objectness_valid))
                self.last_selection_debug["valid_final_count"] = 0
                self.last_selection_debug["objectness_threshold"] = float(self.objectness_threshold)
            lost_action_id = self.select_action_by_lost_target_progress(
                score,
                endstate_w,
                start_pos,
                stamp_sec,
                "target_distance_lost_target_ekf_progress",
            )
            if lost_action_id is not None:
                return lost_action_id
            return fallback_id

        valid_ids = np.flatnonzero(valid)
        distances = np.linalg.norm(selection_end_positions[valid_ids] - target_world.reshape(1, 3), axis=1)
        best_pos = int(np.argmin(distances))
        action_id = int(valid_ids[best_pos])
        target_vec = target_world - start_pos
        target_norm = float(np.linalg.norm(target_vec))
        progress = 0.0
        lateral_error = 0.0
        if target_norm > 1e-6:
            target_dir = target_vec / target_norm
            terminal_vec = selection_end_positions[action_id] - start_pos
            progress = float(np.dot(terminal_vec, target_dir))
            lateral_error = float(np.linalg.norm(terminal_vec - progress * target_dir))
        self.last_selection_debug.update({
            "fallback": False,
            "reason": "target_distance_terminal",
            "valid_score_count": int(np.count_nonzero(score_valid)),
            "valid_final_count": int(valid_ids.size),
            "score_margin": float(self.target_selection_score_margin),
            "target_world": [float(x) for x in target_world],
            "selected_terminal_distance": float(distances[best_pos]),
            "selected_terminal_progress": progress,
            "selected_terminal_lateral_error": lateral_error,
            "paper_preselect_action": int(fallback_id),
        })
        if objectness_valid is not None:
            self.last_selection_debug["valid_objectness_count"] = int(np.count_nonzero(objectness_valid))
            self.last_selection_debug["objectness_threshold"] = float(self.objectness_threshold)
        return action_id

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
        truth_target = self.get_truth_target_world(rospy.Time.now().to_sec(), require_recent=True)
        if truth_target is not None and self.allow_truth_target():
            debug["target_source"] = "truth"
            debug["target_truth_world"] = [float(x) for x in truth_target]
        else:
            debug["target_source"] = "detector"
            if truth_target is not None:
                debug["target_truth_world"] = [float(x) for x in truth_target]
                debug["target_truth_debug_only"] = True
        if objectness_pred is not None:
            obj = objectness_pred.reshape(self.lattice_primitive.traj_num)
            debug["max_objectness"] = float(np.max(obj))
            debug["selected_objectness"] = float(obj[selected_action_id])
        if target_uvd_pred is not None:
            target_uvd = target_uvd_pred.reshape(self.lattice_primitive.traj_num, 3)
            debug["selected_target_uvd"] = [float(x) for x in target_uvd[selected_action_id]]
        if self.last_target_debug:
            debug["target"] = self.last_target_debug
        if self.last_no_target_debug:
            debug["no_target"] = self.last_no_target_debug
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
        now = time.time()
        should_log_interval = (
            self.perf_log_interval >= 0.0
            and (
                self.count == 1
                or self.perf_log_interval == 0.0
                or now - self.last_perf_log_time >= self.perf_log_interval
            )
        )
        if total_time > tolerance:
            rospy.logwarn(f"Warn: Processing time {(time5 - time0) * 1000:.2f} ms exceeds {tolerance:.2f} ms, may cause message lag!")
            print(f"\033[34mCurrent Time Consuming:\033[0m "
                  f"depth-interpolation: \033[32m{1000 * (time1 - time0):.2f} ms\033[0m; "
                  f"data-prepare: \033[32m{1000 * (time2 - time1):.2f} ms\033[0m; "
                  f"network-inference: \033[32m{1000 * (time3 - time2):.2f} ms\033[0m; "
                  f"post-process: \033[32m{1000 * (time4 - time3):.2f} ms\033[0m; "
                  f"visualize-trajectory: \033[32m{1000 * (time5 - time4):.2f} ms\033[0m",
                  flush=True)
        if self.verbose or should_log_interval or (total_time > tolerance):
            print(f"\033[34mAverage Time Consuming:\033[0m "
                  f"depth-interpolation: \033[32m{1000 * self.time_interpolation / self.count:.2f} ms\033[0m; "
                  f"data-prepare: \033[32m{1000 * self.time_prepare / self.count:.2f} ms\033[0m; "
                  f"network-inference: \033[32m{1000 * self.time_forward / self.count:.2f} ms\033[0m; "
                  f"post-process: \033[32m{1000 * self.time_process / self.count:.2f} ms\033[0m; "
                  f"visualize-trajectory: \033[32m{1000 * self.time_visualize / self.count:.2f} ms\033[0m",
                  flush=True)
            self.last_perf_log_time = now

    def warm_up(self):
        depth = torch.zeros((1, self.image_channels, self.height, self.width), dtype=torch.float32, device=self.device)
        obs = torch.zeros((1, int(cfg.get("observation_dim", 6))), dtype=torch.float32, device=self.device)
        obs = self.state_transform.prepare_input(obs)
        output = self.policy(depth, obs)
        endstate_pred = output[0]
        _ = self.state_transform.pred_to_endstate(endstate_pred)


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime_config", "--config_yaml", dest="runtime_config", type=str, default="",
                        help="YAML file with runtime args; command-line args override YAML values")
    parser.add_argument("--use_tensorrt", type=int, default=0, help="use tensorrt or not")
    parser.add_argument("--trial", type=int, default=1, help="trial number")
    parser.add_argument("--epoch", type=int, default=50, help="epoch number")
    parser.add_argument("--weight", type=str, default="/mnt/nvme0n1p5/YOPO/YOPO/saved/YOPO_23/epoch5.pth", help="explicit checkpoint path")
    parser.add_argument("--goal", type=float, nargs=3, default=[0.0, 0.0, 2.0], help="world-frame goal/target prior")
    parser.add_argument("--velocity", type=float, default=None,
                        help="override cfg['velocity'] for runtime lattice speed without editing traj_opt.yaml")
    parser.add_argument("--odom_topic", type=str, default="/LIVO2/imu_propagate")
    parser.add_argument("--depth_topic", type=str, default="/camera/aligned_depth_to_color/image_raw")
    parser.add_argument("--rgb_topic", type=str, default="/camera/color/image_raw")
    parser.add_argument("--rgb_timeout", type=float, default=0.2)
    parser.add_argument("--ctrl_topic", type=str, default="/planning/pos_cmd")
    parser.add_argument("--plan_from_reference", type=int, default=0)
    parser.add_argument("--publish_pos_cmd", type=int, default=1)
    parser.add_argument("--publish_poly_traj", type=int, default=0)
    parser.add_argument("--poly_traj_topic", type=str, default="/planning_cmd/poly_traj")
    parser.add_argument("--pitch_angle_deg", type=float, default=0.0)
    parser.add_argument("--target_camera_Rcl", type=float, nargs=9, default=None,
                        help="optional FAST-LIVO/LiDAR-to-camera optical rotation, row-major")
    parser.add_argument("--target_camera_Pcl", type=float, nargs=3, default=None,
                        help="optional FAST-LIVO/LiDAR-to-camera optical translation")
    parser.add_argument("--selection_mode", type=str, default="paper", choices=["score", "objectness", "hybrid", "paper", "target_distance"])
    parser.add_argument("--objectness_threshold", type=float, default=0.5)
    parser.add_argument("--target_selection_score_margin", type=float, default=1.5,
                        help="target_distance mode keeps candidates with score <= min_score + this margin; negative disables the score gate")
    parser.add_argument("--target_consistency_gate", type=float, default=3.0,
                        help="paper mode rejects detector targets farther than this from EKF prediction; <=0 disables the gate")
    parser.add_argument("--target_nms_distance", type=float, default=1.0,
                        help="paper mode target-space NMS distance in meters; <=0 disables NMS")
    parser.add_argument("--target_timeout", type=float, default=1.0)
    parser.add_argument("--target_truth_topic", type=str, default="")
    parser.add_argument("--target_truth_timeout", type=float, default=1.0)
    parser.add_argument("--target_truth_follow_z", type=float, default=float("nan"),
                        help="optional z override for target truth, useful when the sim publishes the person's ground/root point")
    parser.add_argument("--use_truth_target_in_paper", type=int, default=0,
                        help="paper mode uses detector/EKF target by default; set 1 to let truth drive yaw/prior/hold")
    parser.add_argument("--no_target_behavior", type=str, default="fallback", choices=["fallback", "hold"],
                        help="fallback keeps the old score fallback; hold sends/keeps hover commands when no recent target exists")
    parser.add_argument("--no_target_timeout", type=float, default=1.0,
                        help="seconds without a recent detector/truth target before no_target_behavior takes effect")
    parser.add_argument("--lost_target_progress_enabled", type=int, default=1,
                        help="when current detection is lost but EKF is recent, prefer trajectories progressing toward predicted target")
    parser.add_argument("--lost_target_progress_timeout", type=float, default=1.0,
                        help="seconds since last target measurement during which EKF lost-target progress can still suppress hold")
    parser.add_argument("--lost_target_progress_weight", type=float, default=0.05,
                        help="weight for runtime lost-target progress cost added to predicted score")
    parser.add_argument("--lost_target_progress_cap", type=float, default=8.0,
                        help="max useful horizontal progress toward predicted target in meters")
    parser.add_argument("--lost_target_progress_xy_only", type=int, default=1,
                        help="use horizontal progress only for lost-target runtime selection")
    parser.add_argument("--target_process_noise", type=float, default=2.0)
    parser.add_argument("--target_measurement_noise", type=float, default=0.5)
    parser.add_argument("--use_target_yaw", type=int, default=1)
    parser.add_argument("--max_yaw_rate_deg_s", type=float, default=float("nan"),
                        help="optional runtime override for cfg['max_yaw_rate_deg_s'] in deg/s")
    parser.add_argument("--use_target_prior", type=int, default=1)
    parser.add_argument("--disable_arrive_check", type=int, default=1)
    parser.add_argument("--planning_min_z", type=float, default=float("nan"), help="optional absolute minimum terminal z")
    parser.add_argument("--planning_max_z", type=float, default=float("nan"), help="optional absolute maximum terminal z")
    parser.add_argument("--planning_max_abs_delta_z", type=float, default=float("nan"), help="optional max terminal z change relative to current/start z")
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--perf_log_interval", type=float, default=1.0,
                        help="seconds between average inference timing logs; 0 logs every frame, negative keeps the old verbose/overrun-only behavior")
    parser.add_argument("--visualize", type=int, default=1)
    return parser


def load_runtime_config(path):
    if not path:
        return {}
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.exists(path):
        raise FileNotFoundError("runtime config not found: {}".format(path))
    try:
        import yaml
        with open(path, "r") as f:
            data = yaml.safe_load(f)
    except ImportError:
        from ruamel.yaml import YAML
        with open(path, "r") as f:
            data = YAML(typ="safe").load(f)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError("runtime config must be a YAML mapping: {}".format(path))
    if set(data.keys()) == {"yopo_runtime"}:
        data = data["yopo_runtime"] or {}
    if not isinstance(data, dict):
        raise ValueError("yopo_runtime must be a YAML mapping: {}".format(path))
    return data


def parse_args_with_runtime_config():
    arg_parser = parser()
    probe_args, _ = arg_parser.parse_known_args()
    runtime_config = load_runtime_config(probe_args.runtime_config)
    if runtime_config:
        valid_keys = {action.dest for action in arg_parser._actions}
        unknown = sorted(set(runtime_config.keys()) - valid_keys)
        if unknown:
            raise ValueError("unknown runtime config key(s): {}. Valid keys are argparse option names.".format(unknown))
        arg_parser.set_defaults(**runtime_config)
    args = arg_parser.parse_args()
    if args.runtime_config:
        print("load runtime config from:", os.path.abspath(os.path.expanduser(args.runtime_config)))
    return args


if __name__ == "__main__":
    args = parse_args_with_runtime_config()
    if args.velocity is not None:
        cfg["velocity"] = float(args.velocity)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if args.weight:
        weight = args.weight
    else:
        weight = "yopo_trt.pth" if args.use_tensorrt else base_dir + "/saved/YOPO_{}/epoch{}.pth".format(args.trial, args.epoch)
    print("load weight from:", weight)

    settings = {'use_tensorrt': args.use_tensorrt,
                'goal': args.goal,      # 目标点位置
                'pitch_angle_deg': args.pitch_angle_deg,   # 相机俯仰角(仰为负)
                'target_camera_Rcl': args.target_camera_Rcl,
                'target_camera_Pcl': args.target_camera_Pcl,
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
                'target_selection_score_margin': args.target_selection_score_margin,
                'target_consistency_gate': args.target_consistency_gate,
                'target_nms_distance': args.target_nms_distance,
                'target_timeout': args.target_timeout,
                'target_truth_topic': args.target_truth_topic,
                'target_truth_timeout': args.target_truth_timeout,
                'target_truth_follow_z': args.target_truth_follow_z,
                'use_truth_target_in_paper': bool(args.use_truth_target_in_paper),
                'no_target_behavior': args.no_target_behavior,
                'no_target_timeout': args.no_target_timeout,
                'lost_target_progress_enabled': bool(args.lost_target_progress_enabled),
                'lost_target_progress_timeout': args.lost_target_progress_timeout,
                'lost_target_progress_weight': args.lost_target_progress_weight,
                'lost_target_progress_cap': args.lost_target_progress_cap,
                'lost_target_progress_xy_only': bool(args.lost_target_progress_xy_only),
                'target_process_noise': args.target_process_noise,
                'target_measurement_noise': args.target_measurement_noise,
                'use_target_yaw': bool(args.use_target_yaw),
                'use_target_prior': bool(args.use_target_prior),
                'disable_arrive_check': bool(args.disable_arrive_check),
                'planning_min_z': args.planning_min_z,
                'planning_max_z': args.planning_max_z,
                'planning_max_abs_delta_z': args.planning_max_abs_delta_z,
                'verbose': bool(args.verbose),               # 打印耗时？
                'perf_log_interval': args.perf_log_interval,
                'visualize': bool(args.visualize)               # 可视化所有轨迹？(实飞改为False节省计算)
                }
    if np.isfinite(args.max_yaw_rate_deg_s):
        settings['max_yaw_rate_deg_s'] = args.max_yaw_rate_deg_s
    YopoNet(settings, weight)
