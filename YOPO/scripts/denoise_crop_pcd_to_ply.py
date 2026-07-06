#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def read_pcd_header(path):
    header = []
    with Path(path).open("rb") as f:
        while True:
            line = f.readline()
            if not line:
                raise RuntimeError(f"Bad PCD header: {path}")
            header.append(line.decode("ascii", "ignore").rstrip())
            if line.strip() == b"DATA binary":
                break
        offset = f.tell()
    text = "\n".join(header)
    fields = re.search(r"FIELDS\s+(.+)", text).group(1).split()
    sizes = list(map(int, re.search(r"SIZE\s+(.+)", text).group(1).split()))
    types = re.search(r"TYPE\s+(.+)", text).group(1).split()
    points = int(re.search(r"POINTS\s+(\d+)", text).group(1))
    if any(size != 4 for size in sizes):
        raise RuntimeError(f"Only 4-byte PCD fields are supported: {path}")
    dtype_fields = []
    for name, typ in zip(fields, types):
        if typ == "F":
            dtype_fields.append((name, "<f4"))
        elif typ == "U":
            dtype_fields.append((name, "<u4"))
        elif typ == "I":
            dtype_fields.append((name, "<i4"))
        else:
            raise RuntimeError(f"Unsupported PCD field type {typ!r} in {path}")
    for required in ("x", "y", "z"):
        if required not in fields:
            raise RuntimeError(f"PCD missing {required} field: {path}")
    return offset, points, np.dtype(dtype_fields)


def iter_cropped_xyz(path, bounds, chunk_points):
    offset, points, dtype = read_pcd_header(path)
    cloud = np.memmap(path, mode="r", dtype=dtype, offset=offset, shape=(points,))
    min_bound = np.asarray(bounds[:3], dtype=np.float32)
    max_bound = np.asarray(bounds[3:], dtype=np.float32)
    for start in range(0, points, chunk_points):
        stop = min(points, start + chunk_points)
        part = cloud[start:stop]
        xyz = np.column_stack([part["x"], part["y"], part["z"]]).astype(np.float32, copy=False)
        mask = np.isfinite(xyz).all(axis=1)
        mask &= np.all(xyz >= min_bound, axis=1)
        mask &= np.all(xyz <= max_bound, axis=1)
        if mask.any():
            yield xyz[mask]


def voxel_downsample_stream(path, bounds, voxel_size, chunk_points):
    min_bound = np.asarray(bounds[:3], dtype=np.float32)
    max_bound = np.asarray(bounds[3:], dtype=np.float32)
    dims = np.ceil((max_bound - min_bound) / float(voxel_size)).astype(np.int64) + 1
    if np.prod(dims.astype(np.float64)) > np.iinfo(np.int64).max:
        raise RuntimeError(f"Voxel grid too large: dims={dims.tolist()}")

    chunk_keys = []
    chunk_points_out = []
    cropped_count = 0
    for xyz in iter_cropped_xyz(path, bounds, chunk_points):
        cropped_count += len(xyz)
        ijk = np.floor((xyz - min_bound) / float(voxel_size)).astype(np.int64)
        keys = ijk[:, 0] + ijk[:, 1] * dims[0] + ijk[:, 2] * dims[0] * dims[1]
        unique_keys, first_idx = np.unique(keys, return_index=True)
        chunk_keys.append(unique_keys)
        chunk_points_out.append(xyz[first_idx])

    if not chunk_keys:
        return np.zeros((0, 3), dtype=np.float32), cropped_count

    keys = np.concatenate(chunk_keys)
    points = np.concatenate(chunk_points_out, axis=0)
    _, first_idx = np.unique(keys, return_index=True)
    return points[first_idx].astype(np.float32), cropped_count


def statistical_outlier_filter(points, mean_k, std_ratio):
    if len(points) == 0:
        return points, np.zeros((0,), dtype=bool), {}
    k = min(int(mean_k) + 1, len(points))
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=k, workers=-1)
    mean_dist = distances[:, 1:].mean(axis=1) if k > 1 else np.zeros((len(points),), dtype=np.float32)
    threshold = float(mean_dist.mean() + float(std_ratio) * mean_dist.std())
    keep = mean_dist <= threshold
    stats = {
        "mean_neighbor_distance_mean": float(mean_dist.mean()),
        "mean_neighbor_distance_std": float(mean_dist.std()),
        "threshold": threshold,
    }
    return points[keep].astype(np.float32), keep, stats


def write_xyz_binary_ply(path, xyz, comments):
    xyz = np.asarray(xyz, dtype="<f4")
    safe_comments = [str(comment).encode("ascii", "ignore").decode("ascii") for comment in comments]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        + "".join(f"comment {comment}\n" for comment in safe_comments)
        + f"element vertex {len(xyz)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    ).encode("ascii")
    with Path(path).open("wb") as f:
        f.write(header)
        xyz.tofile(f)


def main():
    parser = argparse.ArgumentParser(description="Crop, voxel downsample, and denoise binary PCD into xyz-only binary PLY.")
    parser.add_argument("--input-pcd", required=True)
    parser.add_argument("--output-ply", required=True)
    parser.add_argument("--stats-json", default=None)
    parser.add_argument("--min-x", type=float, required=True)
    parser.add_argument("--max-x", type=float, required=True)
    parser.add_argument("--min-y", type=float, required=True)
    parser.add_argument("--max-y", type=float, required=True)
    parser.add_argument("--min-z", type=float, required=True)
    parser.add_argument("--max-z", type=float, required=True)
    parser.add_argument("--voxel-size", type=float, default=0.15)
    parser.add_argument("--mean-k", type=int, default=8)
    parser.add_argument("--std-ratio", type=float, default=2.0)
    parser.add_argument("--chunk-points", type=int, default=2_000_000)
    args = parser.parse_args()

    bounds = [args.min_x, args.min_y, args.min_z, args.max_x, args.max_y, args.max_z]
    voxel_points, cropped_count = voxel_downsample_stream(args.input_pcd, bounds, args.voxel_size, args.chunk_points)
    cleaned, keep_mask, sor_stats = statistical_outlier_filter(voxel_points, args.mean_k, args.std_ratio)

    comments = [
        "cropped voxel-downsampled statistical-outlier-filtered from binary PCD",
        f"source={Path(args.input_pcd).name}",
        f"crop=({args.min_x},{args.max_x},{args.min_y},{args.max_y},{args.min_z},{args.max_z})",
        f"voxel_size={args.voxel_size}, mean_k={args.mean_k}, std_ratio={args.std_ratio}",
        f"cropped_points={cropped_count}, voxel_points={len(voxel_points)}, kept_points={len(cleaned)}",
    ]
    write_xyz_binary_ply(args.output_ply, cleaned, comments)

    summary = {
        "input_pcd": str(Path(args.input_pcd).resolve()),
        "output_ply": str(Path(args.output_ply).resolve()),
        "crop": {
            "min_x": args.min_x,
            "max_x": args.max_x,
            "min_y": args.min_y,
            "max_y": args.max_y,
            "min_z": args.min_z,
            "max_z": args.max_z,
        },
        "voxel_size": args.voxel_size,
        "mean_k": args.mean_k,
        "std_ratio": args.std_ratio,
        "cropped_points": int(cropped_count),
        "voxel_points": int(len(voxel_points)),
        "kept_points": int(len(cleaned)),
        "removed_by_sor": int(len(voxel_points) - len(cleaned)),
        "sor": sor_stats,
    }
    if args.stats_json:
        Path(args.stats_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
