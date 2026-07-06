import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.config import cfg
from policy.yopo_network import YopoNetwork


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="saved/YOPO_8/epoch48.pth")
    parser.add_argument("--output_dir", default="run_logs/synthetic_targets_eval")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image_mode", choices=["blank", "depth_blob"], default="depth_blob")
    parser.add_argument("--prior", choices=["exact", "zero", "noisy"], default="exact")
    parser.add_argument("--noise_xy", type=float, default=0.5)
    parser.add_argument("--noise_z", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def camera_intrinsics():
    width = int(cfg["image_width"])
    height = int(cfg["image_height"])
    fx = cfg.get("fx", None)
    fy = cfg.get("fy", None)
    cx = cfg.get("cx", width * 0.5 - 0.5)
    cy = cfg.get("cy", height * 0.5 - 0.5)
    if fx is None:
        fx = width / (2.0 * math.tan(math.radians(float(cfg["horizon_camera_fov"])) * 0.5))
    if fy is None:
        fy = height / (2.0 * math.tan(math.radians(float(cfg["vertical_camera_fov"])) * 0.5))
    return width, height, float(fx), float(fy), float(cx), float(cy)


def target_body_to_uvd(target_body, fx, fy, cx, cy):
    x, y, z = [float(v) for v in target_body]
    u = cx - y * fx / max(x, 1e-6)
    v = cy - z * fy / max(x, 1e-6)
    return np.array([u, v, x], dtype=np.float32)


def uvd_to_target_body(uvd, fx, fy, cx, cy):
    u, v, depth = [float(x) for x in uvd]
    x = depth
    y = -((u - cx) / fx) * depth
    z = -((v - cy) / fy) * depth
    return np.array([x, y, z], dtype=np.float32)


def make_positions():
    positions = []
    for x in (4.0, 6.0, 8.0, 10.0, 12.0):
        positions.append((x, 0.0, 0.0))
        positions.append((x, 0.12 * x, 0.0))
        positions.append((x, -0.12 * x, 0.0))
        positions.append((x, 0.0, 0.08 * x))
        positions.append((x, 0.0, -0.08 * x))
    return np.asarray(positions, dtype=np.float32)


def make_image(target_uvd, image_mode):
    channels = int(cfg.get("image_channels", 1))
    width = int(cfg["image_width"])
    height = int(cfg["image_height"])
    max_depth = float(cfg.get("max_depth", 20.0))
    depth = np.ones((height, width), dtype=np.float32)
    rgb = np.zeros((3, height, width), dtype=np.float32)

    if image_mode == "depth_blob":
        u, v, d = [float(x) for x in target_uvd]
        yy, xx = np.ogrid[:height, :width]
        radius = max(2.0, 0.018 * width)
        mask = (xx - u) ** 2 + (yy - v) ** 2 <= radius ** 2
        depth[mask] = np.clip(d / max_depth, 0.0, 1.0)
        rgb[:, mask] = np.array([[1.0], [0.15], [0.05]], dtype=np.float32)

    if channels == 1:
        return depth[None, :, :]
    if channels == 3:
        return rgb
    if channels == 4:
        return np.concatenate([rgb, depth[None, :, :]], axis=0)
    raise ValueError(f"Unsupported image_channels={channels}")


def select_prior(target_body, mode, rng, noise_xy, noise_z):
    if mode == "exact":
        return target_body.copy()
    if mode == "zero":
        return np.zeros(3, dtype=np.float32)
    noise = np.array(
        [
            rng.normal(0.0, noise_xy),
            rng.normal(0.0, noise_xy),
            rng.normal(0.0, noise_z),
        ],
        dtype=np.float32,
    )
    return (target_body + noise).astype(np.float32)


def summarize(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": None, "median": None, "p95": None, "max": None}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95.0)),
        "max": float(np.max(arr)),
    }


@torch.inference_mode()
def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    width, height, fx, fy, cx, cy = camera_intrinsics()

    device = torch.device(args.device)
    model = YopoNetwork().to(device)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    rows = []
    for idx, target_body in enumerate(make_positions()):
        target_uvd = target_body_to_uvd(target_body, fx, fy, cx, cy)
        if not (0.0 <= target_uvd[0] < width and 0.0 <= target_uvd[1] < height):
            continue

        image_np = make_image(target_uvd, args.image_mode)
        obs_np = np.zeros(6, dtype=np.float32)

        image = torch.from_numpy(image_np[None]).to(device)
        obs = torch.from_numpy(obs_np[None]).to(device)
        obs_norm = model.state_transform.normalize_obs(obs.clone())
        obs_input = model.state_transform.prepare_input(obs_norm)
        output = model(image, obs_input)
        if len(output) != 4:
            raise RuntimeError("Checkpoint does not expose a tracking target head")
        _endstate, _score, objectness_logit, target_raw = output
        objectness = torch.sigmoid(objectness_logit)[0].detach().cpu().numpy()
        pred_uvd = model.state_transform.decode_target(target_raw)[0].detach().cpu().numpy()

        grid_w = width / float(cfg["horizon_num"])
        grid_h = height / float(cfg["vertical_num"])
        true_row = int(np.clip(math.floor(float(target_uvd[1]) / grid_h), 0, int(cfg["vertical_num"]) - 1))
        true_col = int(np.clip(math.floor(float(target_uvd[0]) / grid_w), 0, int(cfg["horizon_num"]) - 1))
        top_flat = int(np.argmax(objectness.reshape(-1)))
        top_row = top_flat // int(cfg["horizon_num"])
        top_col = top_flat % int(cfg["horizon_num"])

        true_cell_uvd = pred_uvd[:, true_row, true_col]
        top1_uvd = pred_uvd[:, top_row, top_col]
        true_cell_body = uvd_to_target_body(true_cell_uvd, fx, fy, cx, cy)
        top1_body = uvd_to_target_body(top1_uvd, fx, fy, cx, cy)
        true_error = float(np.linalg.norm(true_cell_body - target_body))
        top1_error = float(np.linalg.norm(top1_body - target_body))

        rows.append(
            {
                "idx": idx,
                "target_x": float(target_body[0]),
                "target_y": float(target_body[1]),
                "target_z": float(target_body[2]),
                "target_u": float(target_uvd[0]),
                "target_v": float(target_uvd[1]),
                "true_row": true_row,
                "true_col": true_col,
                "top1_row": top_row,
                "top1_col": top_col,
                "top1_grid_match": int(top_row == true_row and top_col == true_col),
                "true_cell_objectness": float(objectness[true_row, true_col]),
                "top1_objectness": float(objectness[top_row, top_col]),
                "true_cell_error_m": true_error,
                "top1_error_m": top1_error,
                "true_cell_pred_x": float(true_cell_body[0]),
                "true_cell_pred_y": float(true_cell_body[1]),
                "true_cell_pred_z": float(true_cell_body[2]),
                "top1_pred_x": float(top1_body[0]),
                "top1_pred_y": float(top1_body[1]),
                "top1_pred_z": float(top1_body[2]),
            }
        )

    csv_path = output_dir / "synthetic_target_errors.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    true_errors = [row["true_cell_error_m"] for row in rows]
    top1_errors = [row["top1_error_m"] for row in rows]
    summary = {
        "checkpoint": str(checkpoint),
        "image_mode": args.image_mode,
        "prior": args.prior,
        "samples": len(rows),
        "intrinsics": {"width": width, "height": height, "fx": fx, "fy": fy, "cx": cx, "cy": cy},
        "top1_grid_acc": float(np.mean([row["top1_grid_match"] for row in rows])) if rows else None,
        "true_cell_error_m": summarize(true_errors),
        "top1_error_m": summarize(top1_errors),
        "csv": str(csv_path),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
