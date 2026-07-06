import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config.config import cfg


class DetectionSampleAssigner:
    def __init__(self):
        self.height = int(cfg["image_height"])
        self.width = int(cfg["image_width"])
        self.vertical_num = int(cfg["vertical_num"])
        self.horizon_num = int(cfg["horizon_num"])
        self.grid_width = self.width / float(self.horizon_num)
        self.grid_height = self.height / float(self.vertical_num)
        self.ignore_neighbor = int(cfg.get("ignore_neighbor_grids", 0))

    def locate_target(self, target_uvd):
        u = target_uvd[:, 0]
        v = target_uvd[:, 1]
        d = target_uvd[:, 2]
        valid = (d > 0) & (u >= 0) & (u < self.width) & (v >= 0) & (v < self.height)
        grid_u = torch.floor(u / self.grid_width).long().clamp(0, self.horizon_num - 1)
        grid_v = torch.floor(v / self.grid_height).long().clamp(0, self.vertical_num - 1)
        return valid, grid_v, grid_u

    def __call__(self, target_uvd):
        device = target_uvd.device
        batch = target_uvd.shape[0]
        pos_mask = torch.zeros(batch, self.vertical_num, self.horizon_num, dtype=torch.bool, device=device)
        neg_mask = torch.ones_like(pos_mask)
        valid, grid_v, grid_u = self.locate_target(target_uvd)

        rows = torch.arange(self.vertical_num, device=device).view(1, self.vertical_num, 1)
        cols = torch.arange(self.horizon_num, device=device).view(1, 1, self.horizon_num)
        close = (rows - grid_v.view(batch, 1, 1)).abs().maximum((cols - grid_u.view(batch, 1, 1)).abs())
        close = (close <= self.ignore_neighbor) & valid.view(batch, 1, 1)

        batch_ids = torch.arange(batch, device=device)
        pos_mask[batch_ids[valid], grid_v[valid], grid_u[valid]] = True
        neg_mask[close] = False
        neg_mask[pos_mask] = False
        ignore_mask = ~(pos_mask | neg_mask)
        return pos_mask, neg_mask, ignore_mask


class DetectionTargetLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.assigner = DetectionSampleAssigner()
        self.max_depth = float(cfg.get("max_depth", 20.0))
        self.grid_width = self.assigner.grid_width
        self.grid_height = self.assigner.grid_height
        self.vertical_num = self.assigner.vertical_num
        self.horizon_num = self.assigner.horizon_num
        self.fx = cfg.get("fx", None)
        self.fy = cfg.get("fy", None)
        self.cx = cfg.get("cx", cfg["image_width"] * 0.5)
        self.cy = cfg.get("cy", cfg["image_height"] * 0.5)
        if self.fx is None:
            self.fx = cfg["image_width"] / (2.0 * math.tan(math.radians(cfg["horizon_camera_fov"]) * 0.5))
        if self.fy is None:
            self.fy = cfg["image_height"] / (2.0 * math.tan(math.radians(cfg["vertical_camera_fov"]) * 0.5))

    def pixel_depth_to_camera(self, uvd, intrinsics=None):
        depth = uvd[..., 2]
        if intrinsics is None:
            fx = self.fx
            fy = self.fy
            cx = self.cx
            cy = self.cy
        else:
            view_shape = [intrinsics.shape[0]] + [1] * (depth.dim() - 1)
            fx = intrinsics[:, 0].view(*view_shape)
            fy = intrinsics[:, 1].view(*view_shape)
            cx = intrinsics[:, 2].view(*view_shape)
            cy = intrinsics[:, 3].view(*view_shape)
        x = depth
        y = (uvd[..., 0] - cx) * depth / fx
        z = (uvd[..., 1] - cy) * depth / fy
        return torch.stack([x, y, z], dim=-1)

    def forward(self, target_uvd_pred, objectness_logit, target_uvd_gt, target_pos_gt=None, target_intrinsics=None):
        pos_mask, neg_mask, ignore_mask = self.assigner(target_uvd_gt)
        zero = objectness_logit.sum() * 0.0

        pos_obj = F.binary_cross_entropy_with_logits(
            objectness_logit[pos_mask], torch.ones_like(objectness_logit[pos_mask])
        ) if pos_mask.any() else zero
        if neg_mask.any():
            neg_logits = objectness_logit[neg_mask]
            neg_losses = F.binary_cross_entropy_with_logits(
                neg_logits, torch.zeros_like(neg_logits), reduction="none"
            )
            neg_obj = neg_losses.mean()
        else:
            neg_logits = None
            neg_losses = None
            neg_obj = zero

        hard_neg_obj = zero
        hard_neg_topk = int(cfg.get("hard_negative_obj_topk", 0))
        hard_neg_weight = float(cfg.get("lambda_hard_negative_obj", 0.0))
        if neg_losses is not None and hard_neg_topk > 0 and hard_neg_weight > 0.0:
            k = min(hard_neg_topk, neg_losses.numel())
            hard_neg_obj = torch.topk(neg_losses, k=k, largest=True).values.mean()

        objectness_loss = (
            pos_obj
            + float(cfg.get("lambda_negative_obj", 0.5)) * neg_obj
            + hard_neg_weight * hard_neg_obj
        )

        if target_pos_gt is None:
            target_pos_gt = self.pixel_depth_to_camera(target_uvd_gt, target_intrinsics)
        pred_pos = self.pixel_depth_to_camera(target_uvd_pred.permute(0, 2, 3, 1), target_intrinsics)
        gt_pos = target_pos_gt[:, None, None, :].expand_as(pred_pos)
        target_loss = F.smooth_l1_loss(pred_pos[pos_mask], gt_pos[pos_mask]) if pos_mask.any() else zero
        return target_loss, objectness_loss, pos_mask, neg_mask, ignore_mask, hard_neg_obj.detach()
