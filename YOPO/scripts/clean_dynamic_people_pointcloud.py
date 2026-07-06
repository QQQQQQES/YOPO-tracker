#!/usr/bin/env python3
import argparse
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


DEFAULT_DATASET_ROOT = "/mnt/nvme0n1p5/YOPO/bag/data/\u6570\u636e\u96c67\u6708"


def read_xyz_binary_ply(path):
    path = Path(path)
    with path.open("rb") as f:
        header = []
        while True:
            line = f.readline()
            if not line:
                raise RuntimeError(f"Bad PLY header: {path}")
            header.append(line.decode("ascii", "ignore").rstrip())
            if line.strip() == b"end_header":
                break

        header_text = "\n".join(header)
        vertex_match = re.search(r"element vertex\s+(\d+)", header_text)
        if not vertex_match:
            raise RuntimeError(f"No vertex count in PLY header: {path}")
        vertex_count = int(vertex_match.group(1))

        props = [line.split()[-1] for line in header if line.startswith("property")]
        dtype = np.dtype([(prop, "<f4") for prop in props])
        arr = np.fromfile(f, dtype=dtype, count=vertex_count)

    for prop in ("x", "y", "z"):
        if prop not in arr.dtype.names:
            raise RuntimeError(f"PLY must contain x/y/z float properties: {path}")
    return np.column_stack([arr["x"], arr["y"], arr["z"]]).astype(np.float32)


def write_xyz_binary_ply(path, xyz, comments):
    path = Path(path)
    xyz = np.asarray(xyz, dtype="<f4")
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        + "".join(f"comment {comment}\n" for comment in comments)
        + f"element vertex {len(xyz)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    ).encode("ascii")

    with path.open("wb") as f:
        f.write(header)
        xyz.tofile(f)


def load_tracks(dataset_root):
    label_files = sorted(Path(dataset_root).glob("scene_*/seq_*/label.npz"))
    if not label_files:
        raise FileNotFoundError(f"No label.npz found under {dataset_root}/scene_*/seq_*")

    poses = []
    targets = []
    for label_file in label_files:
        label = np.load(label_file)
        if "positions" not in label.files:
            raise KeyError(f"{label_file} has no positions field")
        poses.append(label["positions"].astype(np.float32).reshape(-1, 3))

        visible = None
        for key in ("target_visible", "target_valid"):
            if key in label.files:
                visible = label[key].astype(bool).reshape(-1)
                break
        if visible is not None and visible.any() and "target_positions" in label.files:
            target_positions = label["target_positions"].astype(np.float32).reshape(-1, 3)
            targets.append(target_positions[visible])

    poses = np.concatenate(poses, axis=0)
    targets = np.concatenate(targets, axis=0) if targets else np.zeros((0, 3), dtype=np.float32)
    return poses, targets, label_files


def make_cylindrical_mask(points, centers, radius, z_lower, z_upper):
    if len(centers) == 0:
        return np.zeros(len(points), dtype=bool), np.zeros(len(points), dtype=np.float32)

    tree = cKDTree(centers[:, :2])
    xy_dist, nearest_idx = tree.query(points[:, :2], k=1, workers=-1)
    z_rel = points[:, 2] - centers[nearest_idx, 2]
    mask = (xy_dist < radius) & (z_rel > z_lower) & (z_rel < z_upper)
    return mask, z_rel


def nearest_stats(points, queries):
    if len(queries) == 0:
        return {}
    tree = cKDTree(points)
    distances, _ = tree.query(queries, k=1, workers=-1)
    return {
        "min": float(np.min(distances)),
        "p1": float(np.percentile(distances, 1)),
        "p5": float(np.percentile(distances, 5)),
        "p50": float(np.percentile(distances, 50)),
        "p95": float(np.percentile(distances, 95)),
        "p99": float(np.percentile(distances, 99)),
        "max": float(np.max(distances)),
        "lt_0p2_ratio": float(np.mean(distances < 0.2)),
        "lt_0p5_ratio": float(np.mean(distances < 0.5)),
        "lt_1p0_ratio": float(np.mean(distances < 1.0)),
    }


def plot_qc(path, kept, removed, poses, targets, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(40)
    kept_sample = kept[rng.choice(len(kept), min(250000, len(kept)), replace=False)]
    removed_sample = (
        removed[rng.choice(len(removed), min(70000, len(removed)), replace=False)]
        if len(removed)
        else removed
    )

    fig, ax = plt.subplots(figsize=(12, 9), dpi=160)
    ax.scatter(
        kept_sample[:, 0],
        kept_sample[:, 1],
        s=0.1,
        c="0.72",
        alpha=0.25,
        linewidths=0,
        label="kept sample",
    )
    if len(removed_sample):
        ax.scatter(
            removed_sample[:, 0],
            removed_sample[:, 1],
            s=0.8,
            c="red",
            alpha=0.65,
            linewidths=0,
            label="removed dynamic/person band",
        )
    ax.plot(poses[:, 0], poses[:, 1], color="blue", lw=1.0, label="LiDAR/odom path")
    if len(targets):
        step = max(1, len(targets) // 1200)
        ax.scatter(
            targets[::step, 0],
            targets[::step, 1],
            s=5,
            c="lime",
            edgecolors="black",
            linewidths=0.15,
            label="target samples",
        )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Remove dynamic person/lidar-holder trail points from a YOPO real dataset PLY."
    )
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--source-ply", default=None)
    parser.add_argument("--active-ply", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--path-radius", type=float, default=0.65)
    parser.add_argument("--target-radius", type=float, default=0.45)
    parser.add_argument("--z-lower", type=float, default=-0.40)
    parser.add_argument("--z-upper", type=float, default=0.80)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--apply", action="store_true", help="Replace active pointcloud-0.ply with cleaned output.")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    source_ply = (
        Path(args.source_ply)
        if args.source_ply
        else dataset_root / "pointcloud-0.with_dynamic_people_20260704.ply"
    )
    active_ply = Path(args.active_ply) if args.active_ply else dataset_root / "pointcloud-0.ply"
    output_dir = Path(args.output_dir) if args.output_dir else dataset_root
    output_dir.mkdir(parents=True, exist_ok=True)

    if not source_ply.exists():
        raise FileNotFoundError(f"Source PLY does not exist: {source_ply}")

    if args.tag:
        tag = args.tag
    else:
        tag = (
            f"path_r{int(args.path_radius * 100):03d}_"
            f"target_r{int(args.target_radius * 100):03d}_"
            f"zl{int(abs(args.z_lower) * 100):03d}_"
            f"zu{int(abs(args.z_upper) * 100):03d}"
        )

    cleaned_ply = output_dir / f"pointcloud-0.cleaned_{tag}.ply"
    removed_ply = output_dir / f"pointcloud-0.removed_{tag}.ply"
    stats_json = output_dir / f"pointcloud_cleaning_{tag}.json"
    qc_png = output_dir / f"pointcloud_cleaning_{tag}_qc.png"

    points = read_xyz_binary_ply(source_ply)
    poses, targets, label_files = load_tracks(dataset_root)

    path_mask, path_z_rel = make_cylindrical_mask(
        points, poses, args.path_radius, args.z_lower, args.z_upper
    )
    target_mask, target_z_rel = make_cylindrical_mask(
        points, targets, args.target_radius, args.z_lower, args.z_upper
    )
    remove_mask = path_mask | target_mask
    kept = points[~remove_mask]
    removed = points[remove_mask]

    comments = [
        "cleaned dynamic person and lidar-holder trail",
        f"source={source_ply.name}",
        f"path_radius={args.path_radius}, target_radius={args.target_radius}",
        f"z_rel=({args.z_lower}, {args.z_upper})",
        f"removed_total={int(remove_mask.sum())}, removed_path={int(path_mask.sum())}, removed_target={int(target_mask.sum())}",
    ]
    write_xyz_binary_ply(cleaned_ply, kept, comments)
    write_xyz_binary_ply(removed_ply, removed, comments)

    if args.apply:
        shutil.copy2(cleaned_ply, active_ply)

    summary = {
        "dataset_root": str(dataset_root),
        "source_ply": str(source_ply),
        "active_ply": str(active_ply),
        "cleaned_ply": str(cleaned_ply),
        "removed_ply": str(removed_ply),
        "applied_to_active": bool(args.apply),
        "label_count": len(label_files),
        "pose_frames": int(len(poses)),
        "target_frames": int(len(targets)),
        "params": {
            "path_radius_m": args.path_radius,
            "target_radius_m": args.target_radius,
            "z_lower_m": args.z_lower,
            "z_upper_m": args.z_upper,
        },
        "counts": {
            "original_points": int(len(points)),
            "kept_points": int(len(kept)),
            "removed_total": int(remove_mask.sum()),
            "removed_path_mask": int(path_mask.sum()),
            "removed_target_mask": int(target_mask.sum()),
            "removed_overlap": int(path_mask.sum() + target_mask.sum() - remove_mask.sum()),
            "removed_total_ratio": float(remove_mask.mean()),
        },
        "removed_zrel_stats": {
            "path_p5_p50_p95": np.round(
                np.percentile(path_z_rel[path_mask], [5, 50, 95]), 3
            ).tolist()
            if path_mask.any()
            else [],
            "target_p5_p50_p95": np.round(
                np.percentile(target_z_rel[target_mask], [5, 50, 95]), 3
            ).tolist()
            if target_mask.any()
            else [],
        },
        "nearest_after": {
            "pose_nearest_after_m": nearest_stats(kept, poses),
            "target_nearest_after_m": nearest_stats(kept, targets),
        },
    }

    if not args.no_plot:
        plot_qc(
            qc_png,
            kept,
            removed,
            poses,
            targets,
            f"Dynamic/person trail cleaning, z lower {args.z_lower:.2f}m",
        )
        summary["qc_png"] = str(qc_png)

    with stats_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
