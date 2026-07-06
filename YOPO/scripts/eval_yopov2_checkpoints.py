import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from loss.detection_loss import DetectionSampleAssigner, DetectionTargetLoss
from policy.yopo_dataset import YOPODataset
from policy.yopo_network import YopoNetwork


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--output_dir", default="run_logs/checkpoint_eval")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_batches", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def update_sum(stats, key, value, count):
    if count <= 0:
        return
    stats[key] = stats.get(key, 0.0) + float(value) * int(count)
    stats[key + "_n"] = stats.get(key + "_n", 0) + int(count)


def mean_stat(stats, key):
    n = stats.get(key + "_n", 0)
    return stats.get(key, 0.0) / max(1, n)


@torch.inference_mode()
def evaluate_checkpoint(checkpoint, dataloader, device, max_batches):
    model = YopoNetwork().to(device)
    state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    assigner = DetectionSampleAssigner()
    detector = DetectionTargetLoss().to(device)
    stats = {}

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= max_batches:
            break
        image, _pos, _rot, obs_b, _map_id, target_uvd, target_camera, _target_body, target_intrinsics = batch
        image = image.to(device, non_blocking=True)
        obs_b = obs_b.to(device, non_blocking=True)
        target_uvd = target_uvd.to(device, non_blocking=True)
        target_camera = target_camera.to(device, non_blocking=True)
        target_intrinsics = target_intrinsics.to(device, non_blocking=True)

        obs_norm = model.state_transform.normalize_obs(obs_b)
        obs_input = model.state_transform.prepare_input(obs_norm)
        output = model(image, obs_input)
        if len(output) != 4:
            raise RuntimeError(f"{checkpoint} is not a YOPOv2 tracking checkpoint")
        _endstate_pred, _score, objectness_logit, target_raw = output
        objectness = torch.sigmoid(objectness_logit)
        target_uvd_pred = model.state_transform.decode_target(target_raw)

        pos_mask, neg_mask, _ignore_mask = assigner(target_uvd)
        valid_sample = pos_mask.flatten(1).any(dim=1)
        valid_count = int(valid_sample.sum().item())
        if valid_count == 0:
            continue

        flat_obj = objectness.flatten(1)
        flat_pos = pos_mask.flatten(1)
        pos_idx = flat_pos.float().argmax(dim=1)
        top1_idx = flat_obj.argmax(dim=1)
        top3_idx = torch.topk(flat_obj, k=min(3, flat_obj.shape[1]), dim=1).indices

        top1_acc = ((top1_idx == pos_idx) & valid_sample).float().sum().item()
        top3_acc = ((top3_idx == pos_idx[:, None]).any(dim=1) & valid_sample).float().sum().item()
        update_sum(stats, "top1_grid_acc", top1_acc / valid_count, valid_count)
        update_sum(stats, "top3_grid_acc", top3_acc / valid_count, valid_count)

        pos_prob = objectness[pos_mask]
        neg_prob = objectness[neg_mask]
        update_sum(stats, "pos_objectness", pos_prob.mean().item(), pos_prob.numel())
        update_sum(stats, "neg_objectness", neg_prob.mean().item(), neg_prob.numel())

        pred_pos = detector.pixel_depth_to_camera(target_uvd_pred.permute(0, 2, 3, 1), target_intrinsics)
        gt_pos = target_camera[:, None, None, :].expand_as(pred_pos)
        pos_error = torch.linalg.norm(pred_pos[pos_mask] - gt_pos[pos_mask], dim=1)
        update_sum(stats, "true_cell_camera_l2_m", pos_error.mean().item(), pos_error.numel())

        pred_uvd_grid = target_uvd_pred.permute(0, 2, 3, 1)
        gt_uvd = target_uvd[:, None, None, :].expand_as(pred_uvd_grid)
        uv_error = torch.linalg.norm(pred_uvd_grid[pos_mask][:, :2] - gt_uvd[pos_mask][:, :2], dim=1)
        depth_error = (pred_uvd_grid[pos_mask][:, 2] - gt_uvd[pos_mask][:, 2]).abs()
        update_sum(stats, "true_cell_uv_l2_px", uv_error.mean().item(), uv_error.numel())
        update_sum(stats, "true_cell_depth_abs_m", depth_error.mean().item(), depth_error.numel())

        batch_ids = torch.arange(flat_obj.shape[0], device=device)
        top_rows = top1_idx // objectness.shape[2]
        top_cols = top1_idx % objectness.shape[2]
        top_pred_pos = pred_pos[batch_ids, top_rows, top_cols]
        top_error = torch.linalg.norm(top_pred_pos[valid_sample] - target_camera[valid_sample], dim=1)
        update_sum(stats, "top1_camera_l2_m", top_error.mean().item(), top_error.numel())

    return {
        "checkpoint": Path(checkpoint).name,
        "samples": int(stats.get("top1_grid_acc_n", 0)),
        "top1_grid_acc": mean_stat(stats, "top1_grid_acc"),
        "top3_grid_acc": mean_stat(stats, "top3_grid_acc"),
        "pos_objectness": mean_stat(stats, "pos_objectness"),
        "neg_objectness": mean_stat(stats, "neg_objectness"),
        "objectness_margin": mean_stat(stats, "pos_objectness") - mean_stat(stats, "neg_objectness"),
        "true_cell_camera_l2_m": mean_stat(stats, "true_cell_camera_l2_m"),
        "top1_camera_l2_m": mean_stat(stats, "top1_camera_l2_m"),
        "true_cell_uv_l2_px": mean_stat(stats, "true_cell_uv_l2_px"),
        "true_cell_depth_abs_m": mean_stat(stats, "true_cell_depth_abs_m"),
    }


def write_csv(rows, output_dir):
    csv_path = output_dir / "checkpoint_eval_summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def plot_metric(rows, output_dir, metrics, filename, title):
    x = [row["checkpoint"].replace("epoch", "").replace(".pth", "") for row in rows]
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=160)
    for metric in metrics:
        ax.plot(x, [row[metric] for row in rows], marker="o", linewidth=1.8, label=metric)
    ax.set_xlabel("checkpoint epoch")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = output_dir / filename
    fig.savefig(out)
    return out


def main():
    args = parse_args()
    torch.set_num_threads(2)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = YOPODataset(mode="valid")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(args.device == "cuda"),
    )

    rows = []
    for checkpoint in args.checkpoints:
        row = evaluate_checkpoint(Path(checkpoint), dataloader, args.device, args.max_batches)
        rows.append(row)
        print(row)

    csv_path = write_csv(rows, output_dir)
    plot1 = plot_metric(
        rows,
        output_dir,
        ["top1_grid_acc", "top3_grid_acc"],
        "checkpoint_grid_accuracy.png",
        "YOPOv2 Checkpoint Target Grid Accuracy",
    )
    plot2 = plot_metric(
        rows,
        output_dir,
        ["true_cell_camera_l2_m", "top1_camera_l2_m"],
        "checkpoint_target_error.png",
        "YOPOv2 Checkpoint Target Position Error",
    )
    plot3 = plot_metric(
        rows,
        output_dir,
        ["pos_objectness", "neg_objectness", "objectness_margin"],
        "checkpoint_objectness.png",
        "YOPOv2 Checkpoint Objectness Separation",
    )
    print("csv", csv_path)
    print("plot", plot1)
    print("plot", plot2)
    print("plot", plot3)


if __name__ == "__main__":
    main()
