import torch
import numpy as np
from config.config import cfg
from policy.primitive import LatticePrimitive


class StateTransform:
    def __init__(self):
        self.lattice_primitive = LatticePrimitive.get_instance()
        self.goal_length = cfg['goal_length']

    def pred_to_endstate(self, endstate_pred: torch.Tensor) -> torch.Tensor:
        """
            Transform the predicted state to the body frame (Original prediction → Primitive frame → Body frame).
            endstate_pred: [batch; px py pz vx vy vz ax ay az; primitive_v; primitive_h]
            :return [batch; px py pz vx vy vz ax ay az; primitive_v; primitive_h] in body frame
        """
        B, V, H = endstate_pred.shape[0], endstate_pred.shape[2], endstate_pred.shape[3]

        # [B, 9, 3, 5] -> [B, 3, 5, 9] -> [B, 15, 9]
        endstate_pred = endstate_pred.permute(0, 2, 3, 1).reshape(B, V * H, 9)

        # 获取 lattice angle 和 rotation (.flip: 由于lattice和grid的顺序相反)
        yaw, pitch = self.lattice_primitive.getAngleLattice()  # [15]
        yaw = yaw.to(endstate_pred.device)
        pitch = pitch.to(endstate_pred.device)
        yaw = yaw.flip(0)[None, :].expand(B, -1)  # [B, 15]
        pitch = pitch.flip(0)[None, :].expand(B, -1)  # [B, 15]
        Rbp = self.lattice_primitive.getRotation().to(endstate_pred.device).flip(0)  # [15, 3, 3]
        Rbp = Rbp[None, :, :, :].expand(B, -1, -1, -1)  # [B, 15, 3, 3]

        delta_yaw = endstate_pred[:, :, 0] * self.lattice_primitive.yaw_diff  # [B, 15]
        delta_pitch = endstate_pred[:, :, 1] * self.lattice_primitive.pitch_diff
        radio = (endstate_pred[:, :, 2] + 1.0) * self.lattice_primitive.radio_range

        cos_pitch = torch.cos(pitch + delta_pitch)
        endstate_x = cos_pitch * torch.cos(yaw + delta_yaw) * radio
        endstate_y = cos_pitch * torch.sin(yaw + delta_yaw) * radio
        endstate_z = torch.sin(pitch + delta_pitch) * radio
        endstate_p = torch.stack([endstate_x, endstate_y, endstate_z], dim=-1)  # [B, 15, 3]

        # vel / acc
        endstate_vp = endstate_pred[:, :, 3:6] * self.lattice_primitive.vel_max  # [B, 15, 3]
        endstate_ap = endstate_pred[:, :, 6:9] * self.lattice_primitive.acc_max  # [B, 15, 3]

        # v/a 变换到 body frame
        endstate_vb = torch.matmul(Rbp, endstate_vp.unsqueeze(-1)).squeeze(-1)  # [B, 15, 3]
        endstate_ab = torch.matmul(Rbp, endstate_ap.unsqueeze(-1)).squeeze(-1)

        endstate = torch.cat([endstate_p, endstate_vb, endstate_ab], dim=-1)  # [B, 15, 9]

        endstate = endstate.permute(0, 2, 1).reshape(B, 9, V, H)  # [B, 9, 3, 5]
        return endstate

    def pred_to_endstate_cpu(self, endstate_pred: np.ndarray, lattice_id: torch.Tensor) -> np.ndarray:
        """
            Used during test:
            Numpy version of pred_to_endstate() on CPU (used in test, x10 times faster than torch on CUDA)
            :return [B; px py pz vx vy vz ax ay az] in body frame
        """
        delta_yaw = endstate_pred[:, 0] * self.lattice_primitive.yaw_diff
        delta_pitch = endstate_pred[:, 1] * self.lattice_primitive.pitch_diff
        radio = (endstate_pred[:, 2] + 1.0) * self.lattice_primitive.radio_range

        if isinstance(lattice_id, torch.Tensor):
            lattice_id = lattice_id.to(self.lattice_primitive.lattice_angle_node.device)

        yaw, pitch = self.lattice_primitive.getAngleLattice(lattice_id)
        yaw, pitch = yaw.cpu().numpy(), pitch.cpu().numpy()
        endstate_x = np.cos(pitch + delta_pitch) * np.cos(yaw + delta_yaw) * radio
        endstate_y = np.cos(pitch + delta_pitch) * np.sin(yaw + delta_yaw) * radio
        endstate_z = np.sin(pitch + delta_pitch) * radio
        endstate_p = np.stack((endstate_x, endstate_y, endstate_z), axis=1)

        endstate_vp = endstate_pred[:, 3:6] * self.lattice_primitive.vel_max
        endstate_ap = endstate_pred[:, 6:9] * self.lattice_primitive.acc_max

        Rpb = self.lattice_primitive.getRotation(lattice_id).cpu().numpy()
        endstate_vb = np.matmul(Rpb, endstate_vp[:, :, np.newaxis]).squeeze(-1)
        endstate_ab = np.matmul(Rpb, endstate_ap[:, :, np.newaxis]).squeeze(-1)

        return np.concatenate((endstate_p, endstate_vb, endstate_ab), axis=1)


    def decode_target(self, target_raw: torch.Tensor) -> torch.Tensor:
        """
            Decode YOPOv2 target output to image-space [u, v, depth].
            target_raw: [batch; 3; primitive_v; primitive_h]
            return: [batch; 3; primitive_v; primitive_h]
        """
        B, _, V, H = target_raw.shape
        device, dtype = target_raw.device, target_raw.dtype
        grid_w = float(cfg["image_width"]) / float(self.lattice_primitive.horizon_num)
        grid_h = float(cfg["image_height"]) / float(self.lattice_primitive.vertical_num)
        max_depth = float(cfg.get("max_depth", 20.0))

        rows = torch.arange(V, device=device, dtype=dtype).view(1, 1, V, 1)
        cols = torch.arange(H, device=device, dtype=dtype).view(1, 1, 1, H)
        uv_local = torch.sigmoid(target_raw[:, 0:2])
        depth = torch.sigmoid(target_raw[:, 2:3]) * max_depth
        u = (cols + uv_local[:, 0:1]) * grid_w
        v = (rows + uv_local[:, 1:2]) * grid_h
        return torch.cat([u.expand(B, 1, V, H), v.expand(B, 1, V, H), depth], dim=1)

    def decode_target_cpu(self, target_raw: np.ndarray) -> np.ndarray:
        """
            Numpy target decoder used during ROS inference.
            target_raw: [N, 3] in image-grid order.
            return: [N, 3] [u, v, depth].
        """
        target_raw = np.asarray(target_raw, dtype=np.float32)
        grid_w = float(cfg["image_width"]) / float(self.lattice_primitive.horizon_num)
        grid_h = float(cfg["image_height"]) / float(self.lattice_primitive.vertical_num)
        max_depth = float(cfg.get("max_depth", 20.0))
        ids = np.arange(target_raw.shape[0], dtype=np.int64)
        rows = ids // self.lattice_primitive.horizon_num
        cols = ids % self.lattice_primitive.horizon_num
        sigmoid = 1.0 / (1.0 + np.exp(-target_raw))
        u = (cols.astype(np.float32) + sigmoid[:, 0]) * grid_w
        v = (rows.astype(np.float32) + sigmoid[:, 1]) * grid_h
        depth = sigmoid[:, 2] * max_depth
        return np.stack((u, v, depth), axis=1)


    def prepare_input(self, obs):
        """
            Transform the observation to the primitive frame (Body frame → Primitive frame → Body frame).
            obs: [batch; vx, vy, vz, ax, ay, az] in body frame
            :return [batch; vx, vy, vz, ax, ay, az; primitive_v; primitive_h] in primitive frame
        """
        B, N = obs.shape[0], self.lattice_primitive.traj_num
        if obs.shape[1] != 6:
            raise ValueError(f"YOPOv2-Tracker uses 6D state input [v_xyz, a_xyz], got shape {tuple(obs.shape)}")

        # 获取所有 Rbp 并倒序排列 (由于lattice和grid的顺序相反)
        Rbp_all = self.lattice_primitive.getRotation().to(obs.device).flip(0)  # shape: [N, 3, 3]

        obs = obs.view(B, 2, 3)  # [B, 2, 3]

        # 扩展 obs 和 Rbp 到 [B, N, 3, 3]
        obs_exp = obs[:, None, :, :].expand(B, N, 2, 3)
        Rbp_exp = Rbp_all[None, :, :, :].expand(B, N, 3, 3)

        # 执行批量坐标变换
        transformed = torch.matmul(obs_exp, Rbp_exp)  # [B, N, 2, 3]

        transformed_flat = transformed.view(B, N, 6)  # [B, N, 6]
        out = transformed_flat.permute(0, 2, 1).contiguous()  # [B, 6, N]
        out = out.view(B, 6, self.lattice_primitive.vertical_num, self.lattice_primitive.horizon_num)  # [B, 6, V, H]
        return out

    def unnormalize_obs(self, vel_acc):
        vel_acc[:, 0:3] = vel_acc[:, 0:3] * self.lattice_primitive.vel_max
        vel_acc[:, 3:6] = vel_acc[:, 3:6] * self.lattice_primitive.acc_max
        return vel_acc

    def normalize_obs(self, vel_acc):
        if vel_acc.shape[1] != 6:
            raise ValueError(f"YOPOv2-Tracker uses 6D state input [v_xyz, a_xyz], got shape {tuple(vel_acc.shape)}")
        vel_acc[:, 0:3] = vel_acc[:, 0:3] / self.lattice_primitive.vel_max
        vel_acc[:, 3:6] = vel_acc[:, 3:6] / self.lattice_primitive.acc_max
        return vel_acc


def rotate_body2world(rot_wb, pos_b):
    """
    Rotate pos_b from body frame to world frame using quaternion q_wb.
    rot_wb: (..., 3, 3)
    pos_b: (..., 3)
    """
    pos_w = torch.matmul(rot_wb, pos_b.unsqueeze(-1)).squeeze(-1)
    return pos_w


def transform_body2world(rot_wb, t_w, pos_b):
    """
    Transform pos_b from body frame to world frame using quaternion q_wb and t_w.
    rot_wb: (..., 3, 3)
    t_w: (..., 3)
    pos_b: (..., 3)
    """
    return rotate_body2world(rot_wb, pos_b) + t_w


def state_body2world(pos_w, rot_wb, pos_b, vel_b, acc_b):
    pos_b = transform_body2world(rot_wb, pos_w, pos_b)
    vel_b = rotate_body2world(rot_wb, vel_b)
    acc_b = rotate_body2world(rot_wb, acc_b)
    return pos_b, vel_b, acc_b


if __name__ == '__main__':
    CoordTransform = StateTransform()
