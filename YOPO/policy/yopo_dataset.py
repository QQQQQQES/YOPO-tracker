import json
import os, sys
import cv2
import time
import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.transform import Rotation as R
from sklearn.model_selection import train_test_split
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.config import cfg
from config.dataset_utils import (
    resolve_dataset_paths,
    resolve_dataset_eval_modes,
    resolve_dataset_repeat_factors,
    resolve_ply_map_entries,
)


class YOPODataset(Dataset):
    def __init__(self, mode='train', val_ratio=0.1):
        super(YOPODataset, self).__init__()
        # image params
        self.height = int(cfg["image_height"])
        self.width = int(cfg["image_width"])
        # ramdom state: x-direction: log-normal distribution, yz-direction: normal distribution
        self.vel_max = cfg["vel_max_train"]
        self.acc_max = cfg["acc_max_train"]
        self.vx_lognorm_mean = np.log(1 - cfg["vx_mean_unit"])
        self.vx_logmorm_sigma = np.log(cfg["vx_std_unit"])
        self.v_mean = np.array([cfg["vx_mean_unit"], cfg["vy_mean_unit"], cfg["vz_mean_unit"]])
        self.v_std = np.array([cfg["vx_std_unit"], cfg["vy_std_unit"], cfg["vz_std_unit"]])
        self.a_mean = np.array([cfg["ax_mean_unit"], cfg["ay_mean_unit"], cfg["az_mean_unit"]])
        self.a_std = np.array([cfg["ax_std_unit"], cfg["ay_std_unit"], cfg["az_std_unit"]])
        self.goal_length = cfg['goal_length']
        self.goal_pitch_std = cfg["goal_pitch_std"]
        self.goal_yaw_std = cfg["goal_yaw_std"]
        self.image_channels = int(cfg.get("image_channels", 1))
        self.mode = mode
        self.target_prior_augmentation = bool(cfg.get("target_prior_augmentation", False))
        self.target_prior_keep_prob = float(cfg.get("target_prior_keep_prob", 0.4))
        self.target_prior_noise_prob = float(cfg.get("target_prior_noise_prob", 0.25))
        self.target_prior_zero_prob = float(cfg.get("target_prior_zero_prob", 0.2))
        self.target_prior_bootstrap_prob = float(cfg.get("target_prior_bootstrap_prob", 0.15))
        self.target_prior_noise_std_xy = float(cfg.get("target_prior_noise_std_xy", 1.0))
        self.target_prior_noise_std_z = float(cfg.get("target_prior_noise_std_z", 0.3))
        self.target_prior_bootstrap_distance = float(cfg.get("target_prior_bootstrap_distance", self.goal_length))
        self._dataset_intrinsics_applied = False
        if mode == 'train': self.print_data()

        # dataset
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dirs = resolve_dataset_paths(base_dir)
        self.data_dir = self.data_dirs[0]
        self.dataset_repeat_factors = resolve_dataset_repeat_factors(len(self.data_dirs))
        self.dataset_eval_modes = resolve_dataset_eval_modes(len(self.data_dirs))
        if len(self.data_dirs) > 1:
            self.flat_mode = False
            self.temporal_mode = True
            self._load_multi_temporal_dataset(mode, val_ratio)
            return

        self.flat_mode = self._is_flat_dataset(self.data_dir)
        if self.flat_mode:
            self._load_flat_dataset(mode, val_ratio)
            return
        self.temporal_mode = self._is_temporal_dataset(self.data_dir)
        if self.temporal_mode:
            self._load_temporal_dataset(mode, val_ratio)
            return

        data_dir = self.data_dir
        self.img_list, self.map_idx, self.positions, self.quaternions = [], [], np.empty((0, 3), dtype=np.float32), np.empty((0, 4), dtype=np.float32)

        datafolders = [f.path for f in os.scandir(data_dir) if f.is_dir()]
        datafolders.sort(key=lambda x: int(os.path.basename(x)))
        if mode == 'train':
            print("Datafolders:")
            for folder in datafolders:
                print("    ", folder)

        print("Loading", mode, "dataset")
        for data_idx in range(len(datafolders)):
            datafolder = datafolders[data_idx]

            image_file_names = [datafolder + "/" + filename
                                for filename in os.listdir(datafolder)
                                if os.path.splitext(filename)[1] == '.png']
            image_file_names.sort(key=lambda x: int(os.path.basename(x).split('.')[0].split("_")[1]))  # sort by filename to align with the label

            states = np.loadtxt(data_dir + f"/pose-{data_idx}.csv", delimiter=',', skiprows=1).astype(np.float32)
            positions = states[:, 0:3]
            quaternions = states[:, 3:7]

            file_names_train, file_names_val, positions_train, positions_val, quaternions_train, quaternions_val = train_test_split(
                image_file_names, positions, quaternions, test_size=val_ratio, random_state=0)

            if mode == 'train':
                self.img_list.extend(file_names_train)
                self.positions = np.vstack((self.positions, positions_train.astype(np.float32)))
                self.quaternions = np.vstack((self.quaternions, quaternions_train.astype(np.float32)))
                self.map_idx.extend([data_idx] * len(file_names_train))
            elif mode == 'valid':
                self.img_list.extend(file_names_val)
                self.positions = np.vstack((self.positions, positions_val.astype(np.float32)))
                self.quaternions = np.vstack((self.quaternions, quaternions_val.astype(np.float32)))
                self.map_idx.extend([data_idx] * len(file_names_val))
            else:
                raise ValueError(f"Invalid mode {mode}. Choose from 'train', 'valid'.")

        print(f"=============== {mode.capitalize()} Data Summary ===============")
        print(f"{'Images'      :<12} | Count: {len(self.img_list):<3} |  Shape: {self.width},{self.height}")
        print(f"{'Positions'   :<12} | Count: {self.positions.shape[0]:<3} |  Shape: {self.positions.shape[1]}")
        print(f"{'Quaternions' :<12} | Count: {self.quaternions.shape[0]:<3} |  Shape: {self.quaternions.shape[1]}")
        print("==================================================")

    def _is_temporal_dataset(self, data_dir):
        if not os.path.isdir(data_dir):
            return False
        for scene_name in os.listdir(data_dir):
            scene_path = os.path.join(data_dir, scene_name)
            if scene_name.startswith("scene_") and os.path.isdir(scene_path):
                return True
        return False

    def _is_flat_dataset(self, data_dir):
        return os.path.isfile(os.path.join(data_dir, "label.npz"))

    def _load_flat_dataset(self, mode, val_ratio):
        label_file = os.path.join(self.data_dir, "label.npz")
        label = np.load(label_file)
        self._apply_dataset_intrinsics(label)
        frames = len(label["timestamps"])
        indices = np.arange(frames)
        train_idx, val_idx = train_test_split(indices, test_size=val_ratio, random_state=0)
        selected_idx = train_idx if mode == 'train' else val_idx if mode == 'valid' else None
        if selected_idx is None:
            raise ValueError(f"Invalid mode {mode}. Choose from 'train', 'valid'.")

        scene_id = int(label["scene_id"]) if np.asarray(label["scene_id"]).shape == () else int(label["scene_id"][0])
        self.samples = [(self.data_dir, label_file, int(frame_idx), scene_id) for frame_idx in selected_idx]

        print(f"Loading {mode} flat tracking dataset from {self.data_dir}")
        print(f"=============== {mode.capitalize()} Data Summary ===============")
        print(f"{'Frames'      :<12} | Count: {len(self.samples):<6} | Shape: {self.width},{self.height}")
        print(f"{'Channels'    :<12} | Count: {self.image_channels:<6}")
        print("==================================================")

    def _load_temporal_dataset(self, mode, val_ratio):
        selected_seq, split_summary, loaded_frames = self._append_temporal_dataset(
            self.data_dir,
            mode,
            val_ratio,
            map_id_offset=0,
            repeat_factor=1,
            split_single_sequence_frames=True,
        )

        print(f"Loading {mode} temporal tracking dataset from {self.data_dir}")
        print(f"=============== {mode.capitalize()} Data Summary ===============")
        print(f"{'Sequences'   :<12} | Count: {len(selected_seq):<6}")
        print(f"{'Frames'      :<12} | Count: {loaded_frames:<6} | Shape: {self.width},{self.height}")
        print(f"{'Channels'    :<12} | Count: {self.image_channels:<6}")
        print(f"{'Split'       :<12} | {split_summary}")
        print("==================================================")

    def _load_multi_temporal_dataset(self, mode, val_ratio):
        self.samples = []
        dataset_summaries = []
        map_id_offset = 0
        for data_idx, data_dir in enumerate(self.data_dirs):
            if not self._is_temporal_dataset(data_dir):
                raise ValueError(f"dataset_paths only supports temporal scene_*/seq_* roots, got: {data_dir}")
            map_entries = resolve_ply_map_entries(data_dir)
            if not map_entries:
                raise FileNotFoundError(f"No usable pointcloud mapping found for dataset root: {data_dir}")
            eval_mode = self.dataset_eval_modes[data_idx]
            include_dataset = eval_mode == "both" or (mode == "train" and eval_mode == "train_only") or (mode == "valid" and eval_mode == "valid_only")
            if not include_dataset:
                dataset_summaries.append(
                    {
                        "path": data_dir,
                        "sequences": 0,
                        "frames": 0,
                        "repeat": 0,
                        "map_offset": map_id_offset,
                        "map_count": len(map_entries),
                        "split": f"skipped mode={eval_mode}",
                        "eval_mode": eval_mode,
                    }
                )
                map_id_offset += len(map_entries)
                continue
            repeat_factor = self.dataset_repeat_factors[data_idx] if mode == 'train' else 1
            selected_seq, split_summary, loaded_frames = self._append_temporal_dataset(
                data_dir,
                mode,
                val_ratio,
                map_id_offset=map_id_offset,
                repeat_factor=repeat_factor,
                split_single_sequence_frames=(eval_mode == "both"),
            )
            dataset_summaries.append(
                {
                    "path": data_dir,
                    "sequences": len(selected_seq),
                    "frames": loaded_frames,
                    "repeat": repeat_factor,
                    "map_offset": map_id_offset,
                    "map_count": len(map_entries),
                    "split": split_summary,
                    "eval_mode": eval_mode,
                }
            )
            map_id_offset += len(map_entries)

        print(f"Loading {mode} mixed temporal tracking dataset")
        print(f"=============== {mode.capitalize()} Data Summary ===============")
        print(f"{'Datasets'    :<12} | Count: {len(self.data_dirs):<6}")
        print(f"{'Frames'      :<12} | Count: {len(self.samples):<6} | Shape: {self.width},{self.height}")
        print(f"{'Channels'    :<12} | Count: {self.image_channels:<6}")
        for idx, summary in enumerate(dataset_summaries):
            print(
                f"  [{idx}] frames={summary['frames']} seq={summary['sequences']} "
                f"repeat={summary['repeat']} map_offset={summary['map_offset']} "
                f"maps={summary['map_count']} mode={summary['eval_mode']} split={summary['split']}"
            )
            print(f"      {summary['path']}")
        print("==================================================")

    def _append_temporal_dataset(
        self,
        data_dir,
        mode,
        val_ratio,
        map_id_offset=0,
        repeat_factor=1,
        split_single_sequence_frames=False,
    ):
        scene_dirs = [f.path for f in os.scandir(data_dir) if f.is_dir() and f.name.startswith("scene_")]
        scene_dirs.sort()
        seq_dirs = []
        for scene_dir in scene_dirs:
            seq_dirs.extend([f.path for f in os.scandir(scene_dir) if f.is_dir() and f.name.startswith("seq_")])
        seq_dirs.sort()
        if not seq_dirs:
            raise FileNotFoundError(f"No scene_*/seq_* folders found in {data_dir}")

        selected_seq, split_summary = self._split_temporal_sequences(
            data_dir,
            scene_dirs,
            seq_dirs,
            mode,
            val_ratio,
            split_single_sequence_frames=split_single_sequence_frames,
        )
        if selected_seq is None:
            raise ValueError(f"Invalid mode {mode}. Choose from 'train', 'valid'.")

        if not hasattr(self, "samples"):
            self.samples = []
        loaded_frames = 0
        use_frame_split = split_single_sequence_frames and len(seq_dirs) == 1
        for seq_dir in selected_seq:
            label_file = os.path.join(seq_dir, "label.npz")
            if not os.path.isfile(label_file):
                continue
            label = np.load(label_file)
            self._apply_dataset_intrinsics(label)
            frames = len(label["timestamps"])
            scene_id = int(label["scene_id"]) + int(map_id_offset)
            frame_indices = self._split_single_sequence_frame_indices(frames, mode, val_ratio) if use_frame_split else range(frames)
            for _ in range(max(1, int(repeat_factor))):
                for frame_idx in frame_indices:
                    self.samples.append((seq_dir, label_file, frame_idx, scene_id))
                    loaded_frames += 1

        return selected_seq, split_summary, loaded_frames

    def _split_temporal_sequences(self, data_dir, scene_dirs, seq_dirs, mode, val_ratio, split_single_sequence_frames=False):
        split_unit = str(cfg.get("temporal_split_unit", "scene")).lower()
        if len(seq_dirs) < 2:
            selected_seq = list(seq_dirs) if mode in ("train", "valid") else None
            if split_single_sequence_frames:
                return selected_seq, f"unit={split_unit} single-sequence frame-tail split val_ratio={val_ratio}"
            return selected_seq, f"unit={split_unit} single-sequence all-frames mode-specific"

        if split_unit == "sequence":
            train_seq, val_seq = train_test_split(seq_dirs, test_size=val_ratio, random_state=0)
            selected_seq = train_seq if mode == 'train' else val_seq if mode == 'valid' else None
            return selected_seq, "unit=sequence"

        groups = defaultdict(list)
        unity_groups = self._load_unity_scene_groups(data_dir) if split_unit == "unity_scene" else {}
        for scene_dir in scene_dirs:
            scene_id = self._scene_id_from_dir(scene_dir)
            if split_unit == "unity_scene":
                group_id = unity_groups.get(scene_id, scene_id)
            elif split_unit == "scene":
                group_id = scene_id
            else:
                raise ValueError("temporal_split_unit must be one of: sequence, scene, unity_scene")

            scene_seq_dirs = [f.path for f in os.scandir(scene_dir) if f.is_dir() and f.name.startswith("seq_")]
            groups[group_id].extend(sorted(scene_seq_dirs))

        group_ids = sorted([group_id for group_id, paths in groups.items() if paths])
        if len(group_ids) < 2:
            if len(seq_dirs) < 2:
                selected_seq = list(seq_dirs) if mode in ("train", "valid") else None
                if split_single_sequence_frames:
                    return selected_seq, f"unit=sequence single-sequence frame-tail split val_ratio={val_ratio}, requested={split_unit}"
                return selected_seq, f"unit=sequence single-sequence all-frames mode-specific, requested={split_unit}"
            train_seq, val_seq = train_test_split(seq_dirs, test_size=val_ratio, random_state=0)
            selected_seq = train_seq if mode == 'train' else val_seq if mode == 'valid' else None
            return selected_seq, f"unit=sequence fallback, requested={split_unit}"

        train_groups, val_groups = train_test_split(group_ids, test_size=val_ratio, random_state=0)
        selected_groups = train_groups if mode == 'train' else val_groups if mode == 'valid' else None
        if selected_groups is None:
            return None, f"unit={split_unit}"

        selected_seq = []
        for group_id in sorted(selected_groups):
            selected_seq.extend(groups[group_id])
        selected_seq.sort()
        return selected_seq, f"unit={split_unit}, groups={sorted(selected_groups)}"

    def _split_single_sequence_frame_indices(self, frames, mode, val_ratio):
        if frames <= 1:
            return range(frames)
        val_frames = max(1, int(np.ceil(float(frames) * float(val_ratio))))
        val_frames = min(frames - 1, val_frames)
        split_at = frames - val_frames
        if mode == 'train':
            return range(0, split_at)
        if mode == 'valid':
            return range(split_at, frames)
        raise ValueError(f"Invalid mode {mode}. Choose from 'train', 'valid'.")

    def _scene_id_from_dir(self, scene_dir):
        name = os.path.basename(scene_dir)
        try:
            return int(name.split("_", 1)[1])
        except (IndexError, ValueError):
            return name

    def _load_unity_scene_groups(self, data_dir=None):
        data_dir = self.data_dir if data_dir is None else data_dir
        metadata_path = os.path.join(data_dir, "metadata.json")
        if not os.path.isfile(metadata_path):
            return {}
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        groups = {}
        for item in metadata.get("scene_map", []):
            groups[int(item["output_scene_id"])] = int(item["unity_scene_id"])
        return groups

    def _apply_dataset_intrinsics(self, label):
        if self._dataset_intrinsics_applied:
            return
        required = ("fx", "fy", "cx", "cy", "image_width", "image_height")
        if not all(key in label.files for key in required):
            return
        label_width = float(np.asarray(label["image_width"]).item())
        label_height = float(np.asarray(label["image_height"]).item())
        if label_width <= 0.0 or label_height <= 0.0:
            return
        scale_x = float(self.width) / label_width
        scale_y = float(self.height) / label_height
        cfg["fx"] = float(np.asarray(label["fx"]).item()) * scale_x
        cfg["fy"] = float(np.asarray(label["fy"]).item()) * scale_y
        cfg["cx"] = float(np.asarray(label["cx"]).item()) * scale_x
        cfg["cy"] = float(np.asarray(label["cy"]).item()) * scale_y
        self._dataset_intrinsics_applied = True
        print(
            "Applied dataset camera intrinsics: "
            f"fx={cfg['fx']:.3f}, fy={cfg['fy']:.3f}, cx={cfg['cx']:.3f}, cy={cfg['cy']:.3f}"
        )

    def __len__(self):
        if getattr(self, "flat_mode", False):
            return len(self.samples)
        if getattr(self, "temporal_mode", False):
            return len(self.samples)
        return len(self.img_list)

    def __getitem__(self, item):
        if getattr(self, "flat_mode", False):
            return self._getitem_temporal(item)
        if getattr(self, "temporal_mode", False):
            return self._getitem_temporal(item)

        # 1. read the image
        # NOTE: The depth images are normalized from 0–20m to a 0–1 and converted to int16 during data collection.
        image = cv2.imread(self.img_list[item], -1).astype(np.float32)
        image = np.expand_dims(cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_NEAREST) / 65535.0, axis=0)

        # W: world frame; B/b: body frame
        # w: level with the ground but with the same orientation (yaw) as the body frame
        q_wxyz = self.quaternions[item, :]  # q: wxyz
        R_WB = R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
        euler_angles = R_WB.as_euler('ZYX', degrees=False)  # [yaw(z) pitch(y) roll(x)]
        R_Bw = R.from_euler('ZYX', [0, euler_angles[1], euler_angles[2]], degrees=False).inv()

        # 2. get random vel, acc in the direction of the quadrotor
        vel_w, acc_w = self._get_random_state()
        vel_b, acc_b = R_Bw.apply(vel_w), R_Bw.apply(acc_w)

        random_obs = np.hstack((vel_b, acc_b)).astype(np.float32)
        rot_wb = R_WB.as_matrix().astype(np.float32)  # transform to rot_matrix in numpy is faster than using quat in pytorch
        # vel & acc are in body frame, NWU, and no-normalization
        return image, self.positions[item], rot_wb, random_obs, self.map_idx[item]

    def _getitem_temporal(self, item):
        seq_dir, label_file, frame_idx, scene_id = self.samples[item]
        label = np.load(label_file)

        image_name = str(label["image_files"][frame_idx])
        depth_name = str(label["depth_files"][frame_idx])
        image_path = os.path.join(seq_dir, image_name)
        depth_path = os.path.join(seq_dir, depth_name)
        image = self._read_temporal_image(image_path, depth_path)

        q_wxyz = label["quaternions"][frame_idx].astype(np.float32)
        rot_wb = R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_matrix().astype(np.float32)
        pos = label["positions"][frame_idx].astype(np.float32)

        visible = bool(label["target_visible"][frame_idx])
        target_pixel = label["target_pixels"][frame_idx].astype(np.float32)
        label_width = float(np.asarray(label["image_width"]).item()) if "image_width" in label.files else float(self.width)
        label_height = float(np.asarray(label["image_height"]).item()) if "image_height" in label.files else float(self.height)
        fx = float(np.asarray(label["fx"]).item()) if "fx" in label.files else self.width / (2.0 * np.tan(np.deg2rad(cfg["horizon_camera_fov"]) * 0.5))
        fy = float(np.asarray(label["fy"]).item()) if "fy" in label.files else self.height / (2.0 * np.tan(np.deg2rad(cfg["vertical_camera_fov"]) * 0.5))
        cx = float(np.asarray(label["cx"]).item()) if "cx" in label.files else label_width * 0.5
        cy = float(np.asarray(label["cy"]).item()) if "cy" in label.files else label_height * 0.5
        if label_width > 0.0 and label_height > 0.0:
            scale_x = float(self.width) / label_width
            scale_y = float(self.height) / label_height
            target_pixel[0] *= scale_x
            target_pixel[1] *= scale_y
            fx *= scale_x
            fy *= scale_y
            cx *= scale_x
            cy *= scale_y
        target_intrinsics = np.array([fx, fy, cx, cy], dtype=np.float32)
        target_body = label["rendered_relative_positions_body"][frame_idx].astype(np.float32)
        if "target_camera" in label.files:
            target_camera = label["target_camera"][frame_idx].astype(np.float32)
        else:
            target_camera = np.array([target_body[0], -target_body[1], -target_body[2]], dtype=np.float32)
        target_uvd = np.array([target_pixel[0], target_pixel[1], target_camera[0]], dtype=np.float32)
        if not visible:
            target_uvd[:] = 0.0
            target_camera[:] = 0.0

        state_body = label["state_body"][frame_idx].astype(np.float32)
        random_obs = np.hstack((state_body[0:3], state_body[3:6])).astype(np.float32)

        return image, pos, rot_wb, random_obs, scene_id, target_uvd, target_camera, target_body, target_intrinsics

    def _read_temporal_image(self, image_path, depth_path):
        if str(depth_path).endswith(".npy"):
            depth = np.load(depth_path).astype(np.float32)
            depth = np.nan_to_num(depth, nan=0.0, posinf=20.0, neginf=0.0)
            depth = np.clip(depth, 0.0, 20.0) / 20.0
        else:
            depth = cv2.imread(depth_path, -1)
            if depth is None:
                raise FileNotFoundError(depth_path)
            depth = depth.astype(np.float32)
            if depth.max() > 1.5:
                depth = depth / 65535.0
            depth = np.clip(depth, 0.0, 1.0)
        if depth.shape[0] != self.height or depth.shape[1] != self.width:
            depth = cv2.resize(depth, (self.width, self.height), interpolation=cv2.INTER_NEAREST)

        if self.image_channels == 1:
            return depth[None, :, :].astype(np.float32)

        rgb = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if rgb is None:
            raise FileNotFoundError(image_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        if rgb.shape[0] != self.height or rgb.shape[1] != self.width:
            rgb = cv2.resize(rgb, (self.width, self.height), interpolation=cv2.INTER_AREA)
        rgb = rgb.astype(np.float32) / 255.0

        if self.image_channels == 3:
            return rgb.transpose(2, 0, 1).astype(np.float32)
        if self.image_channels == 4:
            return np.concatenate((rgb.transpose(2, 0, 1), depth[None, :, :]), axis=0).astype(np.float32)
        raise ValueError(f"Unsupported image_channels={self.image_channels}. Expected 1, 3, or 4.")

    def _get_random_state(self):
        while True:
            vel = self.vel_max * (self.v_mean + self.v_std * np.random.randn(3))
            right_skewed_vx = -1
            while right_skewed_vx < 0:
                right_skewed_vx = self.vel_max * np.random.lognormal(mean=self.vx_lognorm_mean, sigma=self.vx_logmorm_sigma, size=None)
                right_skewed_vx = -right_skewed_vx + 1.2 * self.vel_max  # * 1.2 to ensure v_max can be sampled
            vel[0] = right_skewed_vx
            if np.linalg.norm(vel) < 1.2 * self.vel_max:  # avoid outliers
                break

        while True:
            acc = self.acc_max * (self.a_mean + self.a_std * np.random.randn(3))
            if np.linalg.norm(acc) < 1.2 * self.acc_max:  # avoid outliers
                break
        return vel, acc

    def _get_random_goal(self):
        goal_pitch_angle = np.random.normal(0.0, self.goal_pitch_std)
        goal_yaw_angle = np.random.normal(0.0, self.goal_yaw_std)
        goal_pitch_angle, goal_yaw_angle = np.radians(goal_pitch_angle), np.radians(goal_yaw_angle)
        goal_w_dir = np.array([np.cos(goal_yaw_angle) * np.cos(goal_pitch_angle),
                               np.sin(goal_yaw_angle) * np.cos(goal_pitch_angle), np.sin(goal_pitch_angle)])
        # 10% probability to generate a nearby goal (× goal_length is actual length)
        random_near = np.random.rand()
        if random_near < 0.1:
            goal_w_dir = random_near * 10 * goal_w_dir
        return self.goal_length * goal_w_dir

    def _augment_target_prior_body(self, target_body, visible):
        target_prior = target_body.astype(np.float32).copy()
        if self.mode != "train" or not self.target_prior_augmentation or not visible:
            return target_prior

        probs = np.asarray(
            [
                self.target_prior_keep_prob,
                self.target_prior_noise_prob,
                self.target_prior_zero_prob,
                self.target_prior_bootstrap_prob,
            ],
            dtype=np.float32,
        )
        prob_sum = float(probs.sum())
        if prob_sum <= 0.0:
            return target_prior
        probs = probs / prob_sum
        mode = int(np.random.choice(4, p=probs))
        if mode == 0:
            return target_prior
        if mode == 1:
            noise = np.array(
                [
                    np.random.normal(0.0, self.target_prior_noise_std_xy),
                    np.random.normal(0.0, self.target_prior_noise_std_xy),
                    np.random.normal(0.0, self.target_prior_noise_std_z),
                ],
                dtype=np.float32,
            )
            return (target_prior + noise).astype(np.float32)
        if mode == 2:
            return np.zeros(3, dtype=np.float32)
        return np.array([self.target_prior_bootstrap_distance, 0.0, 0.0], dtype=np.float32)

    def print_data(self):
        import scipy.stats as stats
        # 计算Vx 5% ~ 95% 区间
        p5 = self.vel_max * np.exp(stats.norm.ppf(0.05, loc=self.vx_lognorm_mean, scale=self.vx_logmorm_sigma))
        p95 = self.vel_max * np.exp(stats.norm.ppf(0.95, loc=self.vx_lognorm_mean, scale=self.vx_logmorm_sigma))

        v_lower = self.vel_max * (self.v_mean - 2 * self.v_std)
        v_upper = self.vel_max * (self.v_mean + 2 * self.v_std)
        v_lower[0] = max(-p95 + 1.2 * self.vel_max, 0)
        v_upper[0] = -p5 + 1.2 * self.vel_max

        a_lower = self.acc_max * (self.a_mean - 2 * self.a_std)
        a_upper = self.acc_max * (self.a_mean + 2 * self.a_std)

        print("----------------- Sampling State --------------------")
        print("| X-Y-Z | Vel 95% Range(m/s)  | Acc 95% Range(m/s2) |")
        print("|-------|---------------------|---------------------|")
        for i in range(3):
            print(f"|  {i:^4} | {v_lower[i]:^9.1f}~{v_upper[i]:^9.1f} |"
                  f" {a_lower[i]:^9.1f}~{a_upper[i]:^9.1f} |")
        print("-----------------------------------------------------")
        print(f"| Goal Pitch 90% (deg)        | {-self.goal_pitch_std * 2:^9.1f}~{self.goal_pitch_std * 2:^9.1f} |")
        print(f"| Goal Yaw   90% (deg)        | {-self.goal_yaw_std * 2:^9.1f}~{self.goal_yaw_std * 2:^9.1f} |")
        print("-----------------------------------------------------")

    def plot_sample_distribution(self):
        import matplotlib.pyplot as plt
        # ===== 采样 =====
        N = 10000
        goals = np.array([self._get_random_goal() for _ in range(N)])
        states = np.array([self._get_random_state() for _ in range(N)])
        vels = np.stack([s[0] for s in states])
        accs = np.stack([s[1] for s in states])

        x, y, z = goals[:, 0], goals[:, 1], goals[:, 2]
        yaw = np.degrees(np.arctan2(y, x))  # 水平角 [-180, 180]
        pitch = np.degrees(np.arctan2(z, np.sqrt(x ** 2 + y ** 2)))  # 垂直角 [-90, 90]

        fig, axs = plt.subplots(3, 3, figsize=(15, 10))

        # Goal方向角分布
        axs[0, 0].hist(yaw, bins=180)
        axs[0, 0].set_title("Goal Yaw Distribution")
        axs[0, 0].set_xlabel("Yaw (deg)")
        axs[0, 0].set_xlim([-60, 60])
        axs[0, 0].grid(True)

        axs[0, 1].hist(pitch, bins=90)
        axs[0, 1].set_title("Goal Pitch Distribution")
        axs[0, 1].set_xlabel("Pitch (deg)")
        axs[0, 1].set_xlim([-60, 60])
        axs[0, 1].grid(True)

        # Goal往图像投影分布(未考虑机体旋转)
        axs[0, 2].scatter(yaw, pitch, s=2, alpha=0.3)
        axs[0, 2].set_title("Goal Distribution in Image")
        axs[0, 2].set_xlabel("Yaw (deg)")
        axs[0, 2].set_ylabel("Pitch (deg)")
        axs[0, 2].set_xlim([-45, 45])
        axs[0, 2].set_ylim([-30, 30])
        axs[0, 2].grid(True)

        # Velocity分布
        for i, name in enumerate(['Vx', 'Vy', 'Vz']):
            axs[1, i].hist(vels[:, i], bins=100)
            axs[1, i].set_title(f"Velocity {name}")
            axs[1, i].grid(True)

        # Acceleration分布
        for i, name in enumerate(['Ax', 'Ay', 'Az']):
            axs[2, i].hist(accs[:, i], bins=100)
            axs[2, i].set_title(f"Acceleration {name}")
            axs[2, i].grid(True)

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    # plot the random sample
    dataset = YOPODataset()
    dataset.plot_sample_distribution()

    # select the best num_workers
    max_workers = os.cpu_count()
    print(f"\n✅ cpu_count = {max_workers}")

    results = []
    for nw in range(0, max_workers + 1):
        data_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=nw)
        start = time.time()
        for i, _ in enumerate(data_loader):
            if i > 50:  # 只测前50个batch
                break
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.time() - start
        results.append((nw, elapsed))
        print(f"num_workers={nw}: {elapsed:.3f}s")

    best = min(results, key=lambda x: x[1])
    print(f"\n✅ 最优 num_workers = {best[0]}, 平均耗时={best[1]:.3f}s")
