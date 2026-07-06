import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from config.config import cfg
from loss.detection_loss import DetectionSampleAssigner, DetectionTargetLoss
from policy.yopo_dataset import YOPODataset
from policy.yopo_network import YopoNetwork


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default="run_logs/checkpoint_vis")
    parser.add_argument("--num_samples", type=int, default=24)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def image_to_rgb(image_chw):
    image = image_chw.detach().cpu().numpy()
    if image.shape[0] >= 3:
        rgb = np.transpose(image[:3], (1, 2, 0))
        return np.clip(rgb, 0.0, 1.0)
    depth = image[0]
    return np.repeat(depth[..., None], 3, axis=2)


def draw_grid(ax, width, height, horizon_num, vertical_num, color="white", alpha=0.35):
    for i in range(1, horizon_num):
        x = i * width / horizon_num
        ax.axvline(x, color=color, linewidth=0.8, alpha=alpha)
    for i in range(1, vertical_num):
        y = i * height / vertical_num
        ax.axhline(y, color=color, linewidth=0.8, alpha=alpha)


def draw_cell(ax, row, col, width, height, horizon_num, vertical_num, color, linewidth=2.0):
    cell_w = width / horizon_num
    cell_h = height / vertical_num
    rect = plt.Rectangle(
        (col * cell_w, row * cell_h),
        cell_w,
        cell_h,
        fill=False,
        edgecolor=color,
        linewidth=linewidth,
    )
    ax.add_patch(rect)


@torch.inference_mode()
def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    dataset = YOPODataset(mode="valid")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = YopoNetwork().to(device)
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    assigner = DetectionSampleAssigner()
    detector = DetectionTargetLoss().to(device)
    width = int(cfg["image_width"])
    height = int(cfg["image_height"])
    horizon_num = int(cfg["horizon_num"])
    vertical_num = int(cfg["vertical_num"])

    samples = []
    for batch in loader:
        image, _pos, _rot, obs_b, _map_id, target_uvd, target_camera, _target_body, target_intrinsics = batch
        image = image.to(device)
        obs_b = obs_b.to(device)
        target_uvd = target_uvd.to(device)
        target_camera = target_camera.to(device)
        target_intrinsics = target_intrinsics.to(device)

        obs_input = model.state_transform.prepare_input(model.state_transform.normalize_obs(obs_b))
        output = model(image, obs_input)
        if len(output) != 4:
            raise RuntimeError(f"{args.checkpoint} is not a YOPOv2 tracking checkpoint")
        _endstate_pred, _score, objectness_logit, target_raw = output
        objectness = torch.sigmoid(objectness_logit)
        target_uvd_pred = model.state_transform.decode_target(target_raw)

        pos_mask, _neg_mask, _ignore_mask = assigner(target_uvd)
        flat_obj = objectness.flatten(1)
        top1_idx = flat_obj.argmax(dim=1)
        top1_row = top1_idx // horizon_num
        top1_col = top1_idx % horizon_num
        true_flat = pos_mask.flatten(1).float().argmax(dim=1)
        true_row = true_flat // horizon_num
        true_col = true_flat % horizon_num

        pred_pos = detector.pixel_depth_to_camera(target_uvd_pred.permute(0, 2, 3, 1), target_intrinsics)
        gt_pos = target_camera
        batch_ids = torch.arange(image.shape[0], device=device)
        pred_top_uvd = target_uvd_pred.permute(0, 2, 3, 1)[batch_ids, top1_row, top1_col]
        pred_top_pos = pred_pos[batch_ids, top1_row, top1_col]
        top_l2 = torch.linalg.norm(pred_top_pos - gt_pos, dim=1)

        for i in range(image.shape[0]):
            if target_uvd[i, 2].item() <= 0:
                continue
            samples.append(
                {
                    "image": image[i].detach().cpu(),
                    "objectness": objectness[i].detach().cpu(),
                    "target_uvd": target_uvd[i].detach().cpu().numpy(),
                    "pred_top_uvd": pred_top_uvd[i].detach().cpu().numpy(),
                    "top1_row": int(top1_row[i].item()),
                    "top1_col": int(top1_col[i].item()),
                    "true_row": int(true_row[i].item()),
                    "true_col": int(true_col[i].item()),
                    "top1_conf": float(flat_obj[i, top1_idx[i]].item()),
                    "true_conf": float(objectness[i][true_row[i], true_col[i]].item()),
                    "top_l2": float(top_l2[i].item()),
                    "top1_hit": bool(top1_idx[i].item() == true_flat[i].item()),
                }
            )
            if len(samples) >= args.num_samples:
                break
        if len(samples) >= args.num_samples:
            break

    rows = math.ceil(len(samples) / 4)
    cols = min(4, max(1, len(samples)))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.0, rows * 2.8), dpi=160)
    axes = np.atleast_1d(axes).reshape(rows, cols)

    for idx, sample in enumerate(samples):
        ax = axes[idx // cols, idx % cols]
        rgb = image_to_rgb(sample["image"])
        ax.imshow(rgb)
        heat = sample["objectness"].numpy()
        heat_img = np.kron(heat, np.ones((height // vertical_num, width // horizon_num)))
        ax.imshow(heat_img, cmap="magma", alpha=0.34, vmin=0.0, vmax=1.0, extent=[0, width, height, 0])
        draw_grid(ax, width, height, horizon_num, vertical_num)
        draw_cell(ax, sample["true_row"], sample["true_col"], width, height, horizon_num, vertical_num, "lime", 2.2)
        draw_cell(ax, sample["top1_row"], sample["top1_col"], width, height, horizon_num, vertical_num, "cyan", 2.0)
        ax.scatter(sample["target_uvd"][0], sample["target_uvd"][1], s=38, c="lime", marker="x", linewidths=2.0)
        ax.scatter(sample["pred_top_uvd"][0], sample["pred_top_uvd"][1], s=24, c="cyan", marker="o", edgecolors="black", linewidths=0.5)
        ax.set_title(
            f"hit={sample['top1_hit']} top={sample['top1_conf']:.2f} "
            f"true={sample['true_conf']:.2f} err={sample['top_l2']:.2f}m",
            fontsize=8,
        )
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.axis("off")

    for idx in range(len(samples), rows * cols):
        axes[idx // cols, idx % cols].axis("off")

    fig.suptitle(f"YOPOv2 target/objectness visualization: {Path(args.checkpoint).name}", fontsize=12)
    fig.tight_layout()
    out_path = output_dir / f"{Path(args.checkpoint).stem}_target_vis.png"
    fig.savefig(out_path)
    print(out_path)


if __name__ == "__main__":
    main()
