import math
import torch as th
import torch.nn as nn
from config.config import cfg
from loss.safety_loss import SafetyLoss
from loss.smoothness_loss import SmoothnessLoss
from loss.guidance_loss import GuidanceLoss
from loss.detection_loss import DetectionTargetLoss


class YOPOLoss(nn.Module):
    def __init__(self):
        """
        Compute the cost: including smoothness, safety, guidance, goal cost, etc.
        Currently, keeping multi-segment polynomial support (not yet verified), but only using a single-segment polynomial (m = 1) for now.
        dp: decision parameters
        df: fixed parameters
        """
        super(YOPOLoss, self).__init__()
        self.sgm_time = cfg["sgm_time"]
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        self._C, self._B, self._L, self._RJ, self._RA = self.qp_generation()
        self._RJ = self._RJ.to(self.device)
        self._RA = self._RA.to(self.device)
        self._L = self._L.to(self.device)
        self.denormalize_weight()
        self.smoothness_loss = SmoothnessLoss(self._RJ, self._RA)
        self.safety_loss = SafetyLoss(self._L) if cfg["wc"] > 0 else None
        self.goal_loss = GuidanceLoss()
        self.detection_loss = DetectionTargetLoss()
        print("------ Actual Loss ------")
        print(f"| {'smooth':<12} = {self.smoothness_weight:6.4f} |")
        print(f"| {'safety':<12} = {self.safety_weight:6.4f} |")
        print(f"| {'goal':<12} = {self.goal_weight:6.4f} |")
        print(f"| {'track_goal':<12} = {self.tracking_goal_weight:6.4f} |")
        print("-------------------------")

    def qp_generation(self):
        # 论文中的映射矩阵
        A = th.zeros((6, 6))
        for i in range(3):
            A[2 * i, i] = math.factorial(i)
            for j in range(i, 6):
                A[2 * i + 1, j] = math.factorial(j) / math.factorial(j - i) * (self.sgm_time ** (j - i))

        # H海森矩阵，对应Jerk
        H = th.zeros((6, 6))
        for i in range(3, 6):
            for j in range(3, 6):
                H[i, j] = i * (i - 1) * (i - 2) * j * (j - 1) * (j - 2) / (i + j - 5) * (self.sgm_time ** (i + j - 5))

        # Q海森矩阵，对应Accel
        Q = th.zeros((6, 6))
        for i in range(2, 6):
            for j in range(2, 6):
                Q[i, j] = (i * (i - 1)) * (j * (j - 1)) / (i + j - 3) * (self.sgm_time ** (i + j - 3))

        return self.stack_opt_dep(A, H, Q)

    def stack_opt_dep(self, A, H, Q):
        Ct = th.zeros((6, 6))
        Ct[[0, 2, 4, 1, 3, 5], [0, 1, 2, 3, 4, 5]] = 1

        _C = th.transpose(Ct, 0, 1)

        B = th.inverse(A)

        B_T = th.transpose(B, 0, 1)

        _L = B @ Ct

        _R_Jerk = _C @ (B_T) @ H @ B @ Ct

        _R_Acc = _C @ (B_T) @ Q @ B @ Ct

        return _C, B, _L, _R_Jerk, _R_Acc

    def denormalize_weight(self):
        """
        Denormalize the cost weight to ensure consistency across different speeds to simplify parameter tuning.
        smoothness cost: time integral of jerk² is used as a smoothness cost.
                         If the speed is scaled by n, the cost is scaled by n⁵ (because jerk * n⁶ and time * 1/n).
        safety cost:     time integral of the distance from trajectory to obstacles.
                         If the speed is scaled by n, the cost is scaled by 1/n (because time * 1/n).
        goal cost:       projection of the trajectory onto goal direction.
                         Independent of speed.
        """
        vel_scale = cfg["vel_max_train"] / 1.0
        self.smoothness_weight = cfg["ws"] / vel_scale ** 5
        self.accele_weight = cfg["wa"] / vel_scale ** 3
        self.safety_weight = cfg["wc"]
        self.goal_weight = cfg["wg"]
        self.tracking_goal_weight = cfg.get("w_tracking_goal", cfg["wg"])

    @staticmethod
    def _project_xy(vec, xy_only):
        if not xy_only:
            return vec
        out = vec.clone()
        out[..., 2] = 0.0
        return out

    def trajectory_positions(self, Df, Dp, eval_points=12):
        batch_size = Dp.shape[0]
        L = self._L.unsqueeze(0).expand(batch_size, -1, -1)
        coefficient = th.zeros(batch_size, 18, device=Dp.device, dtype=Dp.dtype)
        for i in range(3):
            d = th.cat([Df[:, i, :], Dp[:, i, :]], dim=1).unsqueeze(-1)
            coefficient[:, 6 * i: 6 * (i + 1)] = (L @ d).squeeze(-1)

        dt = self.sgm_time / eval_points
        t = th.linspace(dt, self.sgm_time, eval_points, device=Dp.device, dtype=Dp.dtype)
        t_power = th.stack([th.ones_like(t), t, t ** 2, t ** 3, t ** 4, t ** 5], dim=-1)
        x = th.sum(t_power.unsqueeze(0) * coefficient[:, None, 0:6], dim=-1)
        y = th.sum(t_power.unsqueeze(0) * coefficient[:, None, 6:12], dim=-1)
        z = th.sum(t_power.unsqueeze(0) * coefficient[:, None, 12:18], dim=-1)
        return th.stack([x, y, z], dim=-1)

    def tracking_goal_loss(self, Df, Dp, target_pos):
        mode = str(cfg.get("tracking_goal_cost_mode", "progress_cap"))
        terminal_pos = Dp[:, :, 0]
        if mode == "l2_terminal":
            return (terminal_pos - target_pos).pow(2).sum(dim=1)
        if mode != "progress_cap":
            raise ValueError(f"Unsupported tracking_goal_cost_mode: {mode}")

        current_pos = Df[:, :, 0]
        target_vec = self._project_xy(target_pos - current_pos, bool(cfg.get("tracking_progress_xy_only", True)))
        terminal_vec = self._project_xy(terminal_pos - current_pos, bool(cfg.get("tracking_progress_xy_only", True)))
        target_dir = target_vec / target_vec.norm(dim=1, keepdim=True).clamp_min(1e-6)
        progress = (terminal_vec * target_dir).sum(dim=1)
        cap = float(cfg.get("tracking_progress_cap", cfg["goal_length"]))
        return cap - progress.clamp(min=0.0, max=cap)

    def forward(self, state, prediction, goal, map_id, target_pos=None, target_valid=None):
        """
        Args:
            prediction: (batch_size, 3, 3) → [px, py, pz; vx, vy, vz; ax, ay, az] in world frame
            state: (batch_size, 3, 3) → [px, py, pz; vx, vy, vz; ax, ay, az] in world frame
            map_id: (batch_size) which ESDF map to query

        Returns:
            cost: (batch_size) → weighted cost
        """
        # Fixed part: initial pos, vel, acc → (batch_size, 3, 3) [px, vx, ax; py, vy, ay; pz, vz, az]
        Df = state.permute(0, 2, 1)

        # Decision parameters (local frame) → (batch_size, 3, 3) [px, vx, ax; py, vy, ay; pz, vz, az]
        Dp = prediction.permute(0, 2, 1)

        smoothness_cost, acceleration_cost = self.smoothness_loss(Df, Dp)
        if self.safety_loss is None:
            safety_cost = prediction.new_zeros(prediction.shape[0])
        else:
            safety_cost = self.safety_loss(Df, Dp, map_id)
        if target_pos is None:
            goal_cost = self.goal_weight * self.goal_loss(Df, Dp, goal)
        else:
            goal_cost = self.tracking_goal_weight * self.tracking_goal_loss(Df, Dp, target_pos)
            if target_valid is not None:
                goal_cost = goal_cost * target_valid.to(device=goal_cost.device, dtype=goal_cost.dtype)

        return self.smoothness_weight * smoothness_cost, self.safety_weight * safety_cost, goal_cost, self.accele_weight * acceleration_cost
