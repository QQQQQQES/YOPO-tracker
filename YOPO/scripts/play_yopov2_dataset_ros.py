import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import rospy
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Image
from std_msgs.msg import String

from policy.yopov2_dataset import load_yopov2_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--mode", type=str, default="valid", choices=["train", "valid", "test"])
    parser.add_argument("--scene_ids", type=int, nargs="*", default=None)
    parser.add_argument("--rate", type=float, default=10.0)
    parser.add_argument("--max_frames", type=int, default=300)
    parser.add_argument("--visible_only", action="store_true")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--namespace", type=str, default="/yopov2_dataset_eval")
    return parser.parse_args()


class YOPOv2DatasetPlayer:
    def __init__(self, args):
        rospy.init_node("yopov2_dataset_player", anonymous=False)
        self.args = args
        self.cfg, self.cfg_path = load_yopov2_config(args.config)
        data_cfg = self.cfg["dataset"]
        self.root = Path(data_cfg["root"])
        self.width = int(data_cfg["image_width"])
        self.height = int(data_cfg["image_height"])
        self.max_depth_m = float(data_cfg["max_depth_m"])
        self.vertical_num = 3
        self.horizon_num = 5

        namespace = args.namespace.rstrip("/")
        self.rgb_pub = rospy.Publisher(f"{namespace}/rgb_image", Image, queue_size=2)
        self.depth_pub = rospy.Publisher(f"{namespace}/depth_image", Image, queue_size=2)
        self.odom_pub = rospy.Publisher(f"{namespace}/sim/odom", Odometry, queue_size=5)
        self.label_pub = rospy.Publisher(f"{namespace}/label", String, queue_size=10)

        self.samples = self._index_samples()
        if not self.samples:
            raise RuntimeError(f"No dataset frames found under {self.root}")
        rospy.loginfo("YOPOv2 dataset player config: %s", self.cfg_path)
        rospy.loginfo("YOPOv2 dataset player root: %s", self.root)
        rospy.loginfo("YOPOv2 dataset player samples: %d", len(self.samples))
        rospy.loginfo("YOPOv2 dataset player namespace: %s", namespace)

    def _scene_ids(self):
        if self.args.scene_ids:
            return [int(x) for x in self.args.scene_ids]
        key = "valid_scene_ids" if self.args.mode == "valid" else f"{self.args.mode}_scene_ids"
        return [int(x) for x in self.cfg["dataset"][key]]

    def _index_samples(self):
        samples = []
        frames_per_sequence = int(self.cfg["dataset"].get("frames_per_sequence", 5))
        for scene_id in self._scene_ids():
            scene_dir = self.root / f"scene_{scene_id:03d}"
            if not scene_dir.exists():
                continue
            for seq_dir in sorted(scene_dir.glob("seq_*")):
                label_path = seq_dir / "label.npz"
                if not label_path.exists():
                    continue
                labels = np.load(label_path, allow_pickle=True)
                for frame_idx in range(frames_per_sequence):
                    visible = bool(labels["target_visible"][frame_idx]) if "target_visible" in labels.files else False
                    if self.args.visible_only and not visible:
                        continue
                    samples.append((seq_dir, frame_idx, scene_id))
                    if self.args.max_frames > 0 and len(samples) >= self.args.max_frames:
                        return samples
        return samples

    def _make_rgb_msg(self, rgb, stamp):
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = "camera"
        msg.height = int(rgb.shape[0])
        msg.width = int(rgb.shape[1])
        msg.encoding = "rgb8"
        msg.is_bigendian = False
        msg.step = int(rgb.shape[1] * 3)
        msg.data = np.ascontiguousarray(rgb).tobytes()
        return msg

    def _make_depth_msg(self, depth, stamp):
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = "camera"
        msg.height = int(depth.shape[0])
        msg.width = int(depth.shape[1])
        msg.encoding = "32FC1"
        msg.is_bigendian = False
        msg.step = int(depth.shape[1] * 4)
        msg.data = np.ascontiguousarray(depth.astype(np.float32)).tobytes()
        return msg

    def _make_odom_msg(self, labels, frame_idx, stamp):
        msg = Odometry()
        msg.header.stamp = stamp
        msg.header.frame_id = "world"
        msg.child_frame_id = "body"

        pos = np.asarray(labels["drone_positions"][frame_idx], dtype=np.float32)
        quat = np.asarray(labels["drone_quaternions"][frame_idx], dtype=np.float32)
        msg.pose.pose.position.x = float(pos[0])
        msg.pose.pose.position.y = float(pos[1])
        msg.pose.pose.position.z = float(pos[2])
        msg.pose.pose.orientation.x = float(quat[0])
        msg.pose.pose.orientation.y = float(quat[1])
        msg.pose.pose.orientation.z = float(quat[2])
        msg.pose.pose.orientation.w = float(quat[3])

        state_body = np.asarray(labels["state_body"][frame_idx], dtype=np.float32)
        vel_body = state_body[:3] if state_body.shape[0] >= 3 else np.zeros(3, dtype=np.float32)
        vel_world = R.from_quat(quat).as_matrix().dot(vel_body)
        msg.twist.twist.linear.x = float(vel_world[0])
        msg.twist.twist.linear.y = float(vel_world[1])
        msg.twist.twist.linear.z = float(vel_world[2])
        return msg

    def _label_json(self, labels, frame_idx, scene_id, seq_dir, stamp):
        source_width = int(labels["image_width"]) if "image_width" in labels.files else self.width
        source_height = int(labels["image_height"]) if "image_height" in labels.files else self.height
        visible = bool(labels["target_visible"][frame_idx]) if "target_visible" in labels.files else False
        has_target = bool(labels["has_target"][frame_idx]) if "has_target" in labels.files else visible

        target_pixel = np.asarray(labels["target_pixels"][frame_idx], dtype=np.float32)
        target_uv = [-1.0, -1.0]
        cell = [-1, -1]
        if visible and target_pixel[0] >= 0 and target_pixel[1] >= 0:
            u = float(target_pixel[0] * self.width / float(source_width))
            v = float(target_pixel[1] * self.height / float(source_height))
            target_uv = [u, v]
            col = int(np.clip(np.floor(u / (self.width / self.horizon_num)), 0, self.horizon_num - 1))
            row = int(np.clip(np.floor(v / (self.height / self.vertical_num)), 0, self.vertical_num - 1))
            cell = [row, col]

        target_body = np.asarray(labels["rendered_relative_positions_body"][frame_idx], dtype=np.float32)
        target_world = np.asarray(labels["target_world"][frame_idx], dtype=np.float32)
        target_depth = float(labels["target_depths"][frame_idx]) if "target_depths" in labels.files else float(np.linalg.norm(target_body))
        if not np.isfinite(target_depth):
            target_depth = float(np.linalg.norm(target_body))

        return {
            "stamp": stamp.to_sec(),
            "scene_id": int(scene_id),
            "seq_dir": str(seq_dir),
            "seq_id": os.path.basename(str(seq_dir)),
            "frame_idx": int(frame_idx),
            "visible": bool(visible),
            "has_target": bool(has_target),
            "target_uv": target_uv,
            "target_cell": cell,
            "target_depth_m": target_depth,
            "target_body": target_body.astype(float).tolist(),
            "target_world": target_world.astype(float).tolist(),
        }

    def publish_sample(self, seq_dir, frame_idx, scene_id):
        labels = np.load(seq_dir / "label.npz", allow_pickle=True)
        stamp = rospy.Time.now()

        rgb_path = seq_dir / str(labels["image_files"][frame_idx])
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb is None:
            raise FileNotFoundError(f"Cannot read RGB image: {rgb_path}")
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        if rgb.shape[:2] != (self.height, self.width):
            rgb = cv2.resize(rgb, (self.width, self.height), interpolation=cv2.INTER_AREA)

        depth_path = seq_dir / str(labels["depth_files"][frame_idx])
        depth = np.load(depth_path).astype(np.float32) if depth_path.suffix == ".npy" else cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32)
        if depth.shape != (self.height, self.width):
            depth = cv2.resize(depth, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        depth = np.nan_to_num(depth, nan=self.max_depth_m, posinf=self.max_depth_m, neginf=0.0)

        self.odom_pub.publish(self._make_odom_msg(labels, frame_idx, stamp))
        self.rgb_pub.publish(self._make_rgb_msg(rgb, stamp))
        self.depth_pub.publish(self._make_depth_msg(depth, stamp))
        self.label_pub.publish(json.dumps(self._label_json(labels, frame_idx, scene_id, seq_dir, stamp), separators=(",", ":")))

    def spin(self):
        rate = rospy.Rate(self.args.rate)
        idx = 0
        while not rospy.is_shutdown():
            if idx >= len(self.samples):
                if not self.args.loop:
                    rospy.loginfo("YOPOv2 dataset player finished")
                    break
                idx = 0
            self.publish_sample(*self.samples[idx])
            idx += 1
            rate.sleep()


if __name__ == "__main__":
    YOPOv2DatasetPlayer(parse_args()).spin()
