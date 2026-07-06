"""
Training Strategy
supervised learning, imitation learning, testing, rollout
"""
import os
import time
import atexit
import numpy as np
import torch
from torch.nn import functional as F
from rich.progress import Progress
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from config.config import cfg
from loss.loss_function import YOPOLoss
from policy.yopo_network import YopoNetwork
from policy.yopo_dataset import YOPODataset
from policy.state_transform import *


class YopoTrainer:
    def __init__(
            self,
            learning_rate=0.001,
            batch_size=32,
            loss_weight=[],
            tensorboard_path=None,
            checkpoint_path=None,
            save_on_exit=False,
            max_train_steps=None,
            max_eval_steps=None,
    ):
        self.batch_size = batch_size
        self.max_grad_norm = 0.1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_weight = loss_weight
        self.max_train_steps = max_train_steps
        self.max_eval_steps = max_eval_steps
        if save_on_exit: self._exit_func = atexit.register(self.save_model)
        # logger
        self.progress_log = Progress()
        self.tensorboard_path = self.get_next_log_path(tensorboard_path)
        self.tensorboard_log = SummaryWriter(log_dir=self.tensorboard_path)
        # params
        self.traj_num = cfg['traj_num']

        # network
        print("Loading network...")
        self.policy = YopoNetwork()
        self.policy = self.policy.to(self.device)
        try:
            state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            self.policy.load_state_dict(state_dict)
            print("Checkpoint ", checkpoint_path, " loaded successfully")
        except FileNotFoundError:
            print("Training from scratch")

        # optimizer
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=learning_rate, fused=(self.device.type == "cuda"))
        print("Network Loaded! Loading Dataset...")

        # dataset (you can adjust num_workers according to your training speed)
        num_workers = int(cfg.get("dataloader_num_workers", 0))
        train_dataset = YOPODataset(mode='train')
        valid_dataset = YOPODataset(mode='valid')
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                           num_workers=num_workers, pin_memory=(self.device.type == "cuda"))
        self.val_dataloader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False,
                                         num_workers=num_workers, pin_memory=(self.device.type == "cuda"))
        print("Dataset Loaded!")

        # Loss reads camera intrinsics from cfg, so construct it after the dataset
        # has had a chance to apply label.npz intrinsics.
        self.yopo_loss = YOPOLoss()

    def train(self, epoch, save_interval=None):
        with self.progress_log:
            total_progress = self.progress_log.add_task("Training", total=epoch)
            for self.epoch_i in range(epoch):
                self.policy.train()
                self.train_one_epoch(self.epoch_i, total_progress)
                self.policy.eval()
                self.eval_one_epoch(self.epoch_i)
                if save_interval is not None and (self.epoch_i + 1) % save_interval == 0:
                    self.progress_log.console.log("Saving model...")
                    policy_path = self.tensorboard_path + "/epoch{}.pth".format(self.epoch_i + 1, 0)
                    torch.save(self.policy.state_dict(), policy_path)
            self.progress_log.console.log("Train YOPO Finish!")
            self.progress_log.remove_task(total_progress)

    def train_one_epoch(self, epoch: int, total_progress):
        train_steps = self._step_limit(len(self.train_dataloader), self.max_train_steps)
        one_epoch_progress = self.progress_log.add_task(f"Epoch: {epoch}", total=train_steps)
        inspect_interval = max(1, train_steps // 16)
        traj_losses, score_losses, smooth_losses, safety_losses, goal_losses, acc_losses, start_time = [], [], [], [], [], [], time.time()
        target_losses, objectness_losses, hard_objectness_losses = [], [], []
        positive_traj_losses, negative_traj_losses, lost_traj_losses, positive_cell_ratios, lost_cell_ratios = [], [], [], [], []
        processed_steps = 0
        for step, batch in enumerate(self.train_dataloader):  # obs: body frame
            if processed_steps >= train_steps:
                break
            depth = batch[0]
            if depth.shape[0] != self.batch_size:  continue  # batch size == number of env

            self.optimizer.zero_grad()

            trajectory_loss, score_loss, smooth_cost, safety_cost, goal_cost, acc_cost, loss_detail = self.forward_and_compute_loss(*batch)

            loss = self.loss_weight[0] * trajectory_loss + self.loss_weight[1] * score_loss

            # Optimize the policy
            loss.backward()
            self.optimizer.step()

            traj_losses.append(self.loss_weight[0] * trajectory_loss.item())
            score_losses.append(self.loss_weight[1] * score_loss.item())
            smooth_losses.append(self.loss_weight[0] * smooth_cost.item())
            safety_losses.append(self.loss_weight[0] * safety_cost.item())
            goal_losses.append(self.loss_weight[0] * goal_cost.item())
            acc_losses.append(self.loss_weight[0] * acc_cost.item())
            target_losses.append(self.loss_weight[0] * loss_detail["target_loss"].item())
            objectness_losses.append(self.loss_weight[0] * loss_detail["objectness_loss"].item())
            hard_objectness_losses.append(self.loss_weight[0] * loss_detail["hard_objectness_loss"].item())
            positive_traj_losses.append(self.loss_weight[0] * loss_detail["positive_traj_loss"].item())
            negative_traj_losses.append(self.loss_weight[0] * loss_detail["negative_traj_loss"].item())
            lost_traj_losses.append(self.loss_weight[0] * loss_detail["lost_traj_loss"].item())
            positive_cell_ratios.append(loss_detail["positive_cell_ratio"].item())
            lost_cell_ratios.append(loss_detail["lost_cell_ratio"].item())

            if step % inspect_interval == inspect_interval - 1:
                batch_fps = inspect_interval / (time.time() - start_time)
                self.progress_log.console.log(f"Epoch: {epoch}, Traj Loss: {np.mean(traj_losses):.3g}, "
                                              f"Score Loss: {np.mean(score_losses):.3g} "
                                              f"Batch FPS: {batch_fps:.3g}")
                self.tensorboard_log.add_scalar("Train/TrajLoss", np.mean(traj_losses), epoch * len(self.train_dataloader) + step)
                self.tensorboard_log.add_scalar("Train/ScoreLoss", np.mean(score_losses), epoch * len(self.train_dataloader) + step)
                self.tensorboard_log.add_scalar("Detail/SmoothLoss", np.mean(smooth_losses), epoch * len(self.train_dataloader) + step)
                self.tensorboard_log.add_scalar("Detail/SafetyLoss", np.mean(safety_losses), epoch * len(self.train_dataloader) + step)
                self.tensorboard_log.add_scalar("Detail/GoalLoss", np.mean(goal_losses), epoch * len(self.train_dataloader) + step)
                self.tensorboard_log.add_scalar("Detail/AccelLoss", np.mean(acc_losses), epoch * len(self.train_dataloader) + step)
                self.tensorboard_log.add_scalar("Detail/TargetLoss", np.mean(target_losses), epoch * len(self.train_dataloader) + step)
                self.tensorboard_log.add_scalar("Detail/ObjectnessLoss", np.mean(objectness_losses), epoch * len(self.train_dataloader) + step)
                self.tensorboard_log.add_scalar("Detail/HardObjectnessLoss", np.mean(hard_objectness_losses), epoch * len(self.train_dataloader) + step)
                self.tensorboard_log.add_scalar("Detail/PositiveTrajLoss", np.mean(positive_traj_losses), epoch * len(self.train_dataloader) + step)
                self.tensorboard_log.add_scalar("Detail/NegativeTrajLoss", np.mean(negative_traj_losses), epoch * len(self.train_dataloader) + step)
                self.tensorboard_log.add_scalar("Detail/LostTrajLoss", np.mean(lost_traj_losses), epoch * len(self.train_dataloader) + step)
                self.tensorboard_log.add_scalar("Detail/PositiveCellRatio", np.mean(positive_cell_ratios), epoch * len(self.train_dataloader) + step)
                self.tensorboard_log.add_scalar("Detail/LostCellRatio", np.mean(lost_cell_ratios), epoch * len(self.train_dataloader) + step)
                traj_losses, score_losses, smooth_losses, safety_losses, goal_losses, acc_losses, start_time = [], [], [], [], [], [], time.time()
                target_losses, objectness_losses, hard_objectness_losses = [], [], []
                positive_traj_losses, negative_traj_losses, lost_traj_losses, positive_cell_ratios, lost_cell_ratios = [], [], [], [], []

            processed_steps += 1
            self.progress_log.update(one_epoch_progress, advance=1)
            self.progress_log.update(total_progress, advance=1 / train_steps)

        self.progress_log.remove_task(one_epoch_progress)

    @torch.inference_mode()
    def eval_one_epoch(self, epoch: int):
        eval_steps = self._step_limit(len(self.val_dataloader), self.max_eval_steps)
        one_epoch_progress = self.progress_log.add_task(f"Eval: {epoch}", total=eval_steps)
        traj_losses, score_losses = [], []
        smooth_losses, safety_losses, goal_losses, acc_losses = [], [], [], []
        target_losses, objectness_losses, hard_objectness_losses = [], [], []
        positive_traj_losses, negative_traj_losses, lost_traj_losses, positive_cell_ratios, lost_cell_ratios = [], [], [], [], []
        processed_steps = 0
        for step, batch in enumerate(self.val_dataloader):  # obs: body frame
            if processed_steps >= eval_steps:
                break
            depth = batch[0]
            if depth.shape[0] != self.batch_size:  continue  # batch size == num of env

            trajectory_loss, score_loss, smooth_cost, safety_cost, goal_cost, acc_cost, loss_detail = self.forward_and_compute_loss(*batch)

            traj_losses.append(self.loss_weight[0] * trajectory_loss.item())
            score_losses.append(self.loss_weight[1] * score_loss.item())
            smooth_losses.append(self.loss_weight[0] * smooth_cost.item())
            safety_losses.append(self.loss_weight[0] * safety_cost.item())
            goal_losses.append(self.loss_weight[0] * goal_cost.item())
            acc_losses.append(self.loss_weight[0] * acc_cost.item())
            target_losses.append(self.loss_weight[0] * loss_detail["target_loss"].item())
            objectness_losses.append(self.loss_weight[0] * loss_detail["objectness_loss"].item())
            hard_objectness_losses.append(self.loss_weight[0] * loss_detail["hard_objectness_loss"].item())
            positive_traj_losses.append(self.loss_weight[0] * loss_detail["positive_traj_loss"].item())
            negative_traj_losses.append(self.loss_weight[0] * loss_detail["negative_traj_loss"].item())
            lost_traj_losses.append(self.loss_weight[0] * loss_detail["lost_traj_loss"].item())
            positive_cell_ratios.append(loss_detail["positive_cell_ratio"].item())
            lost_cell_ratios.append(loss_detail["lost_cell_ratio"].item())
            processed_steps += 1
            self.progress_log.update(one_epoch_progress, advance=1)

        if traj_losses:
            self.progress_log.console.log(f"Eval: {epoch}, Traj Loss: {np.mean(traj_losses):.3g}, Score Loss: {np.mean(score_losses):.3g} ")
            self.tensorboard_log.add_scalar("Eval/TrajLoss", np.mean(traj_losses), epoch)
            self.tensorboard_log.add_scalar("Eval/ScoreLoss", np.mean(score_losses), epoch)
            self.tensorboard_log.add_scalar("EvalDetail/SmoothLoss", np.mean(smooth_losses), epoch)
            self.tensorboard_log.add_scalar("EvalDetail/SafetyLoss", np.mean(safety_losses), epoch)
            self.tensorboard_log.add_scalar("EvalDetail/GoalLoss", np.mean(goal_losses), epoch)
            self.tensorboard_log.add_scalar("EvalDetail/AccelLoss", np.mean(acc_losses), epoch)
            self.tensorboard_log.add_scalar("EvalDetail/TargetLoss", np.mean(target_losses), epoch)
            self.tensorboard_log.add_scalar("EvalDetail/ObjectnessLoss", np.mean(objectness_losses), epoch)
            self.tensorboard_log.add_scalar("EvalDetail/HardObjectnessLoss", np.mean(hard_objectness_losses), epoch)
            self.tensorboard_log.add_scalar("EvalDetail/PositiveTrajLoss", np.mean(positive_traj_losses), epoch)
            self.tensorboard_log.add_scalar("EvalDetail/NegativeTrajLoss", np.mean(negative_traj_losses), epoch)
            self.tensorboard_log.add_scalar("EvalDetail/LostTrajLoss", np.mean(lost_traj_losses), epoch)
            self.tensorboard_log.add_scalar("EvalDetail/PositiveCellRatio", np.mean(positive_cell_ratios), epoch)
            self.tensorboard_log.add_scalar("EvalDetail/LostCellRatio", np.mean(lost_cell_ratios), epoch)
        else:
            self.progress_log.console.log(f"Eval: {epoch}, no full batches to evaluate.")
        self.progress_log.remove_task(one_epoch_progress)

    def _step_limit(self, dataloader_len, configured_limit):
        if configured_limit is None or int(configured_limit) <= 0:
            return max(1, dataloader_len)
        return max(1, min(int(configured_limit), dataloader_len))

    def forward_and_compute_loss(self, depth, pos, rot, obs_b, map_id, target_uvd=None, target_pos_camera=None, target_pos_body=None, target_intrinsics=None):
        depth, pos, rot, obs_b, map_id = [x.to(self.device) for x in [depth, pos, rot, obs_b, map_id]]
        tracking_batch = target_uvd is not None
        if tracking_batch:
            target_uvd = target_uvd.to(self.device)
            target_pos_camera = target_pos_camera.to(self.device)
            target_pos_body = target_pos_body.to(self.device)
            if target_intrinsics is not None:
                target_intrinsics = target_intrinsics.to(self.device)

        # 1. pre-process
        start_vel_w = rotate_body2world(rot, obs_b[:, 0:3])
        start_acc_w = rotate_body2world(rot, obs_b[:, 3:6])
        forward_goal_b = obs_b.new_zeros((obs_b.shape[0], 3))
        forward_goal_b[:, 0] = float(cfg["goal_length"])
        goal_w = rotate_body2world(rot, forward_goal_b) + pos
        start_state_w = torch.stack([pos, start_vel_w, start_acc_w], dim=1)

        # 2. forward propagation
        obs_norm = self.policy.state_transform.normalize_obs(obs_b)
        obs_input = self.policy.state_transform.prepare_input(obs_norm)
        output = self.policy(depth, obs_input)
        if len(output) == 2:
            endstate_pred, score = output
            objectness_logit, target_uvd_pred = None, None
        else:
            endstate_pred, score, objectness_logit, target_raw = output
            target_uvd_pred = self.policy.state_transform.decode_target(target_raw)
        endstate = self.policy.state_transform.pred_to_endstate(endstate_pred)

        # 3. post-process [B, V, H, 9] -> [B*V*H, 9]
        endstate_flat = endstate.permute(0, 2, 3, 1).reshape(self.batch_size * self.traj_num, 9)
        score_flat = score.reshape(self.batch_size * self.traj_num)

        pos_expanded = pos.repeat_interleave(self.traj_num, dim=0)  # [B*V*H, 3]
        rot_expanded = rot.repeat_interleave(self.traj_num, dim=0)  # [B*V*H, 3, 3]
        start_state_w = start_state_w.repeat_interleave(self.traj_num, dim=0)  # [B*V*H, 3, 3]
        goal_w = goal_w.repeat_interleave(self.traj_num, dim=0)  # [B*V*H, 3]

        # [B*V*H, 3] [B*V*H, 3] [B*V*H, 3]
        end_pos_w, end_vel_w, end_acc_w = state_body2world(
            pos_expanded, rot_expanded,
            endstate_flat[:, 0:3],
            endstate_flat[:, 3:6],
            endstate_flat[:, 6:9]
        )
        # [B*V*H, 3, 3]: [px, py, pz; vx, vy, vz; ax, ay, az]
        end_state_w = torch.stack([end_pos_w, end_vel_w, end_acc_w], dim=1)

        if tracking_batch:
            target_pos_world = rotate_body2world(rot, target_pos_body) + pos
            target_pos_world = target_pos_world.repeat_interleave(self.traj_num, dim=0)
            target_valid = (target_uvd[:, 2] > 0.0).repeat_interleave(self.traj_num)
            smooth_cost, safety_cost, goal_cost, acc_cost = self.yopo_loss(
                start_state_w,
                end_state_w,
                goal_w,
                map_id,
                target_pos=target_pos_world,
                target_valid=target_valid,
            )
        else:
            smooth_cost, safety_cost, goal_cost, acc_cost = self.yopo_loss(start_state_w, end_state_w, goal_w, map_id)

        base_cost = smooth_cost + safety_cost + acc_cost
        tracking_use_goal_cost = bool(cfg.get("tracking_use_goal_cost", False))
        tracking_score_include_goal_cost = bool(cfg.get("tracking_score_include_goal_cost", False))
        if tracking_batch:
            full_cost = base_cost + goal_cost if tracking_use_goal_cost else base_cost
            score_label = full_cost.detach() if tracking_score_include_goal_cost else base_cost.detach()
        else:
            full_cost = base_cost + goal_cost
            score_label = full_cost.detach()
        trajectory_loss = full_cost.mean()
        score_weight = None
        zero = trajectory_loss * 0.0
        goal_cost_log = goal_cost.mean()
        loss_detail = {
            "target_loss": zero,
            "objectness_loss": zero,
            "hard_objectness_loss": zero,
            "positive_traj_loss": trajectory_loss,
            "negative_traj_loss": zero,
            "lost_traj_loss": zero,
            "positive_cell_ratio": zero,
            "lost_cell_ratio": zero,
        }

        if tracking_batch and objectness_logit is not None and target_uvd_pred is not None:
            target_loss, objectness_loss, pos_mask, _neg_mask, _ignore_mask, hard_objectness_loss = self.yopo_loss.detection_loss(
                target_uvd_pred,
                objectness_logit,
                target_uvd,
                target_pos_camera,
                target_intrinsics,
            )
            target_valid_frame = target_uvd[:, 2] > 0.0
            visible_grid_mask = target_valid_frame.repeat_interleave(self.traj_num)
            lost_grid_mask = ~visible_grid_mask
            positive_grid_mask = pos_mask.reshape(-1) & visible_grid_mask
            visible_non_positive = visible_grid_mask & ~positive_grid_mask

            traj_cost = base_cost.clone()
            goal_cost_for_label = base_cost + goal_cost if tracking_use_goal_cost else base_cost
            if tracking_use_goal_cost and positive_grid_mask.any():
                traj_cost[positive_grid_mask] = goal_cost_for_label[positive_grid_mask]
            non_positive_goal_weight = float(cfg.get("tracking_non_positive_goal_weight", 0.0))
            if tracking_use_goal_cost and non_positive_goal_weight > 0.0 and visible_non_positive.any():
                traj_cost[visible_non_positive] = (
                    base_cost[visible_non_positive]
                    + non_positive_goal_weight * goal_cost[visible_non_positive]
                )
            non_positive_length_weight = float(cfg.get("tracking_non_positive_length_weight", 0.0))
            if non_positive_length_weight > 0.0 and visible_non_positive.any():
                non_positive_length_cost = self._terminal_length_cost(
                    start_state_w,
                    end_state_w,
                    weight_key="tracking_non_positive_length_weight",
                    cap_key="tracking_non_positive_length_cap",
                    xy_only_key="tracking_non_positive_length_xy_only",
                )
                traj_cost[visible_non_positive] = (
                    traj_cost[visible_non_positive]
                    + non_positive_length_cost[visible_non_positive]
                )

            target_body_expanded = target_pos_body.repeat_interleave(self.traj_num, dim=0)
            lost_direction_valid = self._lost_forward_direction_valid(target_body_expanded)
            lost_forward_mask = lost_grid_mask & lost_direction_valid
            lost_forward_cost = self._lost_forward_cost(start_state_w, end_state_w, rot_expanded, target_body_expanded)
            tracking_use_lost_forward_cost = bool(cfg.get("tracking_use_lost_forward_cost", False))
            if tracking_use_lost_forward_cost and lost_forward_mask.any():
                traj_cost[lost_forward_mask] = base_cost[lost_forward_mask] + lost_forward_cost[lost_forward_mask]

            positive_loss = traj_cost[positive_grid_mask].mean() if positive_grid_mask.any() else zero
            negative_loss = traj_cost[visible_non_positive].mean() if visible_non_positive.any() else zero
            lost_loss = traj_cost[lost_grid_mask].mean() if lost_grid_mask.any() else zero
            positive_weight = float(cfg.get("lambda_positive_traj", 1.0))
            negative_weight = float(cfg.get("lambda_negative_traj", 0.2))
            lost_weight = float(cfg.get("lambda_lost_traj", 0.2))
            trajectory_loss = positive_weight * positive_loss + negative_weight * negative_loss + lost_weight * lost_loss
            trajectory_loss = trajectory_loss + float(cfg.get("w_target", 1.0)) * target_loss + float(cfg.get("w_objectness", 1.0)) * objectness_loss
            score_label = base_cost.detach().clone()
            if tracking_score_include_goal_cost and tracking_use_goal_cost and positive_grid_mask.any():
                score_label[positive_grid_mask] = traj_cost.detach()[positive_grid_mask]
            if (
                bool(cfg.get("tracking_score_include_non_positive_goal_cost", tracking_score_include_goal_cost))
                and tracking_use_goal_cost
                and non_positive_goal_weight > 0.0
                and visible_non_positive.any()
            ):
                score_label[visible_non_positive] = traj_cost.detach()[visible_non_positive]
            if (
                bool(cfg.get("tracking_score_include_non_positive_length_cost", False))
                and non_positive_length_weight > 0.0
                and visible_non_positive.any()
            ):
                score_label[visible_non_positive] = traj_cost.detach()[visible_non_positive]
            if (
                bool(cfg.get("tracking_score_include_lost_forward_cost", False))
                and tracking_use_lost_forward_cost
                and lost_forward_mask.any()
            ):
                score_label[lost_forward_mask] = traj_cost.detach()[lost_forward_mask]
            score_weight = torch.full_like(score_label, negative_weight)
            score_weight[positive_grid_mask] = positive_weight
            score_weight[lost_grid_mask] = lost_weight
            goal_cost_log = goal_cost[positive_grid_mask].mean() if positive_grid_mask.any() else zero
            loss_detail = {
                "target_loss": target_loss.detach(),
                "objectness_loss": objectness_loss.detach(),
                "hard_objectness_loss": hard_objectness_loss.detach(),
                "positive_traj_loss": positive_loss.detach(),
                "negative_traj_loss": negative_loss.detach(),
                "lost_traj_loss": lost_loss.detach(),
                "positive_cell_ratio": positive_grid_mask.float().mean().detach(),
                "lost_cell_ratio": lost_grid_mask.float().mean().detach(),
                "lost_direction_cell_ratio": lost_forward_mask.float().mean().detach(),
            }

        score_loss_raw = F.smooth_l1_loss(score_flat, score_label, reduction="none")
        if score_weight is not None:
            score_loss = (score_loss_raw * score_weight).sum() / score_weight.sum().clamp_min(1e-6)
        else:
            score_loss = score_loss_raw.mean()
        return trajectory_loss, score_loss, smooth_cost.mean(), safety_cost.mean(), goal_cost_log, acc_cost.mean(), loss_detail

    def _lost_forward_direction_valid(self, lost_target_body):
        if lost_target_body is None:
            return None
        direction = lost_target_body
        if bool(cfg.get("lost_forward_xy_only", True)):
            direction = direction.clone()
            direction[:, 2] = 0.0
        min_norm = float(cfg.get("lost_forward_min_target_norm", 1e-3))
        return direction.norm(dim=1) > min_norm

    def _terminal_length_cost(self, start_state_w, end_state_w, weight_key, cap_key, xy_only_key):
        start_pos = start_state_w[:, 0, :]
        terminal_pos = end_state_w[:, 0, :]
        terminal_vec = terminal_pos - start_pos
        if bool(cfg.get(xy_only_key, True)):
            terminal_vec = terminal_vec.clone()
            terminal_vec[:, 2] = 0.0
        length = terminal_vec.norm(dim=1)
        cap = float(cfg.get(cap_key, cfg.get("tracking_progress_cap", cfg["goal_length"])))
        length_cost = cap - length.clamp(min=0.0, max=cap)
        return float(cfg.get(weight_key, 0.0)) * length_cost

    def _lost_forward_cost(self, start_state_w, end_state_w, rot_wb, lost_target_body=None):
        start_pos = start_state_w[:, 0, :]
        terminal_pos = end_state_w[:, 0, :]
        forward_b = torch.zeros_like(start_pos)
        forward_b[:, 0] = 1.0
        body_forward_dir = rotate_body2world(rot_wb, forward_b)

        direction_mode = str(cfg.get("lost_forward_direction", "target")).lower()
        if direction_mode == "target" and lost_target_body is not None:
            forward_dir = rotate_body2world(rot_wb, lost_target_body)
            target_norm = forward_dir.norm(dim=1, keepdim=True)
            fallback_mask = target_norm.squeeze(1) < 1e-6
            if fallback_mask.any():
                forward_dir = forward_dir.clone()
                forward_dir[fallback_mask] = body_forward_dir[fallback_mask]
        elif direction_mode in ("body", "body_forward", "forward"):
            forward_dir = body_forward_dir
        else:
            raise ValueError(f"Unsupported lost_forward_direction: {direction_mode}")

        if bool(cfg.get("lost_forward_xy_only", True)):
            forward_dir[:, 2] = 0.0
            xy_norm = forward_dir.norm(dim=1, keepdim=True)
            fallback_mask = xy_norm.squeeze(1) < 1e-6
            if fallback_mask.any():
                forward_dir = forward_dir.clone()
                forward_dir[fallback_mask] = body_forward_dir[fallback_mask]
                forward_dir[:, 2] = 0.0
        forward_dir = forward_dir / forward_dir.norm(dim=1, keepdim=True).clamp_min(1e-6)

        terminal_vec = terminal_pos - start_pos
        if bool(cfg.get("lost_forward_xy_only", True)):
            terminal_vec = terminal_vec.clone()
            terminal_vec[:, 2] = 0.0
        progress = (terminal_vec * forward_dir).sum(dim=1)
        cap = float(cfg.get("lost_forward_progress_cap", cfg.get("tracking_progress_cap", cfg["goal_length"])))
        forward_cost = cap - progress.clamp(min=0.0, max=cap)
        return float(cfg.get("w_lost_forward", 0.0)) * forward_cost

    def save_model(self):
        if hasattr(self, "epoch_i"):
            self.progress_log.console.log("Saving model...")
            policy_path = self.tensorboard_path + "/epoch{}.pth".format(self.epoch_i + 1, 0)
            torch.save(self.policy.state_dict(), policy_path)
            atexit.unregister(self._exit_func)

    def get_next_log_path(self, base_path):
        nums = [int(name.split("_")[1])
                for name in os.listdir(base_path)
                if os.path.isdir(os.path.join(base_path, name)) and name.startswith("YOPO_") and name.split("_")[1].isdigit()]
        next_n = max(nums, default=-1) + 1
        next_path = os.path.join(base_path, f"YOPO_{next_n}")
        os.makedirs(next_path, exist_ok=False)
        print("record tensorboard log to ", next_path)
        return next_path
