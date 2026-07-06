#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import numpy as np


def scalar(label, key, default=None):
    if key not in label.files:
        return default
    return float(np.asarray(label[key]).item())


def intrinsics(label):
    width = scalar(label, "image_width", 160.0)
    height = scalar(label, "image_height", 90.0)
    fx = scalar(label, "fx", None)
    fy = scalar(label, "fy", None)
    cx = scalar(label, "cx", None)
    cy = scalar(label, "cy", None)
    hfov = scalar(label, "horizon_fov", 90.0)
    vfov = scalar(label, "vertical_fov", 60.0)
    if fx is None:
        fx = width / (2.0 * math.tan(math.radians(hfov) * 0.5))
    if fy is None:
        fy = height / (2.0 * math.tan(math.radians(vfov) * 0.5))
    if cx is None:
        cx = width * 0.5
    if cy is None:
        cy = height * 0.5
    return width, height, fx, fy, cx, cy


def metric(values):
    if not values:
        return None
    arr = np.concatenate(values).astype(np.float64)
    if arr.size == 0:
        return None
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def check_label(label_path):
    out = {
        "labels": 1,
        "frames": 0,
        "visible_frames": 0,
        "has_target_camera_labels": 0,
        "missing_target_camera_labels": 0,
        "projection_pixel_error_px": [],
        "uvd_to_camera_error_m": [],
        "camera_body_axis_error_m": [],
        "camera_body_rcl_pcl_error_m": [],
        "runtime_uvd_to_body_error_m": [],
        "runtime_uvd_to_body_rcl_pcl_error_m": [],
        "no_signflip_body_error_m": [],
    }
    try:
        label = np.load(label_path)
    except Exception as exc:
        out["load_errors"] = [f"{label_path}: {exc}"]
        return out

    if "target_pixels" not in label.files or "target_visible" not in label.files:
        out["schema_errors"] = [f"{label_path}: missing target_pixels or target_visible"]
        return out

    target_pixels = label["target_pixels"].astype(np.float64)
    visible = label["target_visible"].astype(bool)
    out["frames"] = int(target_pixels.shape[0])
    out["visible_frames"] = int(np.count_nonzero(visible))
    if out["visible_frames"] == 0:
        return out

    width, height, fx, fy, cx, cy = intrinsics(label)
    body = None
    if "rendered_relative_positions_body" in label.files:
        body = label["rendered_relative_positions_body"].astype(np.float64)

    if "target_camera" in label.files:
        camera = label["target_camera"].astype(np.float64)
        out["has_target_camera_labels"] = 1
    elif body is not None:
        camera = np.stack((body[:, 0], -body[:, 1], -body[:, 2]), axis=1)
        out["missing_target_camera_labels"] = 1
    else:
        out["schema_errors"] = [f"{label_path}: missing target_camera and rendered_relative_positions_body"]
        return out

    valid = visible & np.isfinite(target_pixels).all(axis=1) & np.isfinite(camera).all(axis=1)
    valid &= camera[:, 0] > 1e-6
    valid &= target_pixels[:, 0] >= 0.0
    valid &= target_pixels[:, 0] < width
    valid &= target_pixels[:, 1] >= 0.0
    valid &= target_pixels[:, 1] < height
    if not np.any(valid):
        return out

    cam = camera[valid]
    pix = target_pixels[valid]
    proj_u = cx + cam[:, 1] * fx / cam[:, 0]
    proj_v = cy + cam[:, 2] * fy / cam[:, 0]
    proj_err = np.linalg.norm(np.stack((proj_u, proj_v), axis=1) - pix, axis=1)
    out["projection_pixel_error_px"].append(proj_err)

    cam_from_uvd = np.stack(
        (
            cam[:, 0],
            (pix[:, 0] - cx) * cam[:, 0] / fx,
            (pix[:, 1] - cy) * cam[:, 0] / fy,
        ),
        axis=1,
    )
    out["uvd_to_camera_error_m"].append(np.linalg.norm(cam_from_uvd - cam, axis=1))

    if body is not None:
        body_valid = body[valid]
        cam_from_body_zero_pitch = np.stack(
            (body_valid[:, 0], -body_valid[:, 1], -body_valid[:, 2]),
            axis=1,
        )
        out["camera_body_axis_error_m"].append(np.linalg.norm(cam_from_body_zero_pitch - cam, axis=1))

        if "Rcl" in label.files and "Pcl" in label.files:
            Rcl = np.asarray(label["Rcl"], dtype=np.float64).reshape(3, 3)
            Pcl = np.asarray(label["Pcl"], dtype=np.float64).reshape(3)
            optical_from_body = body_valid.dot(Rcl.T) + Pcl.reshape(1, 3)
            packed_from_body = np.stack(
                (optical_from_body[:, 2], optical_from_body[:, 0], optical_from_body[:, 1]),
                axis=1,
            )
            out["camera_body_rcl_pcl_error_m"].append(np.linalg.norm(packed_from_body - cam, axis=1))

        runtime_body_from_uvd = np.stack(
            (
                cam[:, 0],
                -((pix[:, 0] - cx) * cam[:, 0] / fx),
                -((pix[:, 1] - cy) * cam[:, 0] / fy),
            ),
            axis=1,
        )
        no_signflip_body = np.stack(
            (
                cam[:, 0],
                (pix[:, 0] - cx) * cam[:, 0] / fx,
                (pix[:, 1] - cy) * cam[:, 0] / fy,
            ),
            axis=1,
        )
        out["runtime_uvd_to_body_error_m"].append(np.linalg.norm(runtime_body_from_uvd - body_valid, axis=1))
        if "Rcl" in label.files and "Pcl" in label.files:
            optical_from_uvd = np.stack(
                (
                    (pix[:, 0] - cx) * cam[:, 0] / fx,
                    (pix[:, 1] - cy) * cam[:, 0] / fy,
                    cam[:, 0],
                ),
                axis=1,
            )
            runtime_body_rcl = (optical_from_uvd - Pcl.reshape(1, 3)).dot(Rcl)
            out["runtime_uvd_to_body_rcl_pcl_error_m"].append(np.linalg.norm(runtime_body_rcl - body_valid, axis=1))
        out["no_signflip_body_error_m"].append(np.linalg.norm(no_signflip_body - body_valid, axis=1))
    return out


def merge_stats(stats):
    merged = {
        "labels": 0,
        "frames": 0,
        "visible_frames": 0,
        "has_target_camera_labels": 0,
        "missing_target_camera_labels": 0,
        "load_errors": [],
        "schema_errors": [],
        "projection_pixel_error_px": [],
        "uvd_to_camera_error_m": [],
        "camera_body_axis_error_m": [],
        "camera_body_rcl_pcl_error_m": [],
        "runtime_uvd_to_body_error_m": [],
        "runtime_uvd_to_body_rcl_pcl_error_m": [],
        "no_signflip_body_error_m": [],
    }
    for item in stats:
        for key in ("labels", "frames", "visible_frames", "has_target_camera_labels", "missing_target_camera_labels"):
            merged[key] += int(item.get(key, 0))
        for key in ("load_errors", "schema_errors"):
            merged[key].extend(item.get(key, []))
        for key in (
            "projection_pixel_error_px",
            "uvd_to_camera_error_m",
            "camera_body_axis_error_m",
            "camera_body_rcl_pcl_error_m",
            "runtime_uvd_to_body_error_m",
            "runtime_uvd_to_body_rcl_pcl_error_m",
            "no_signflip_body_error_m",
        ):
            merged[key].extend(item.get(key, []))

    metrics = {
        key: metric(merged[key])
        for key in (
            "projection_pixel_error_px",
            "uvd_to_camera_error_m",
            "camera_body_axis_error_m",
            "camera_body_rcl_pcl_error_m",
            "runtime_uvd_to_body_error_m",
            "runtime_uvd_to_body_rcl_pcl_error_m",
            "no_signflip_body_error_m",
        )
    }
    return {
        "labels": merged["labels"],
        "frames": merged["frames"],
        "visible_frames": merged["visible_frames"],
        "has_target_camera_labels": merged["has_target_camera_labels"],
        "missing_target_camera_labels": merged["missing_target_camera_labels"],
        "load_errors": merged["load_errors"],
        "schema_errors": merged["schema_errors"],
        "metrics": metrics,
    }


def status_for(summary, pixel_tol, metric_tol):
    if summary["load_errors"] or summary["schema_errors"]:
        return "fail"
    if summary["visible_frames"] == 0:
        return "no_visible_targets"

    metrics = summary["metrics"]
    projection = metrics["projection_pixel_error_px"]
    camera_axis = metrics["camera_body_rcl_pcl_error_m"] or metrics["camera_body_axis_error_m"]
    runtime_body = metrics["runtime_uvd_to_body_rcl_pcl_error_m"] or metrics["runtime_uvd_to_body_error_m"]
    if projection is not None and projection["p95"] > pixel_tol:
        return "fail"
    if camera_axis is not None and camera_axis["p95"] > metric_tol:
        return "fail"
    if runtime_body is not None and runtime_body["p95"] > metric_tol:
        return "fail"
    return "pass"


def main():
    parser = argparse.ArgumentParser(
        description="Check YOPOv2 temporal label coordinate signs between pixel/depth, camera, and body frames."
    )
    parser.add_argument("roots", nargs="+", help="dataset roots to scan")
    parser.add_argument("--max-labels-per-root", type=int, default=0, help="0 means scan every label.npz")
    parser.add_argument("--pixel-p95-tol", type=float, default=3.0)
    parser.add_argument("--metric-p95-tol", type=float, default=0.25)
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()

    report = {
        "convention": {
            "camera": "[x=depth, y=image_right, z=image_down]",
            "body_or_network": "[x=forward, y=left, z=up]",
            "zero_pitch_body_to_camera": "camera=[body_x, -body_y, -body_z]",
            "rcl_pcl_body_to_camera": "optical=Rcl*body+Pcl; camera=[optical_z, optical_x, optical_y]",
            "runtime_uvd_to_body": "body=[depth, -(u-cx)*depth/fx, -(v-cy)*depth/fy]",
            "runtime_uvd_to_body_rcl_pcl": "optical=[right, down, depth]; body=Rcl.T*(optical-Pcl)",
        },
        "roots": {},
    }

    overall_fail = False
    for root_arg in args.roots:
        root = Path(root_arg)
        labels = sorted(root.rglob("label.npz"))
        if args.max_labels_per_root and len(labels) > args.max_labels_per_root:
            labels = labels[: args.max_labels_per_root]
        stats = [check_label(path) for path in labels]
        summary = merge_stats(stats)
        summary["status"] = status_for(summary, args.pixel_p95_tol, args.metric_p95_tol)
        summary["scanned_label_files"] = len(labels)
        report["roots"][str(root)] = summary
        if summary["status"] == "fail":
            overall_fail = True

    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(text + "\n", encoding="utf-8")
    raise SystemExit(1 if overall_fail else 0)


if __name__ == "__main__":
    main()
