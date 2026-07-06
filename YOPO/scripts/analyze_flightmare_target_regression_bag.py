#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os

import rosbag


def pose_to_row(msg, bag_time):
    stamp = msg.header.stamp.to_sec() if msg.header.stamp else bag_time
    return {
        "t": float(stamp),
        "bag_t": float(bag_time),
        "x": float(msg.pose.position.x),
        "y": float(msg.pose.position.y),
        "z": float(msg.pose.position.z),
    }


def nearest_by_time(rows, t, start_idx=0):
    if not rows:
        return None, start_idx
    idx = min(max(start_idx, 0), len(rows) - 1)
    while idx + 1 < len(rows) and abs(rows[idx + 1]["t"] - t) <= abs(rows[idx]["t"] - t):
        idx += 1
    return rows[idx], idx


def finite_stats(values):
    values = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    n = len(values)
    p95_idx = min(n - 1, int(math.ceil(0.95 * n)) - 1)
    return {
        "count": n,
        "mean": sum(values) / n,
        "median": values[n // 2] if n % 2 else 0.5 * (values[n // 2 - 1] + values[n // 2]),
        "p95": values[p95_idx],
        "max": values[-1],
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("bag")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--truth_topic", default="/flightmare/yopo/target_world")
    parser.add_argument("--pred_topic", default="/yopo_net/target_world")
    parser.add_argument("--debug_topic", default="/yopo_net/tracking_debug")
    parser.add_argument("--max_dt", type=float, default=0.1)
    parser.add_argument("--warmup", type=float, default=0.0)
    parser.add_argument("--truth_z_override", type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    truth_rows = []
    pred_rows = []
    debug_rows = []
    with rosbag.Bag(args.bag) as bag:
        start_time = None
        for topic, msg, t in bag.read_messages(topics=[args.truth_topic, args.pred_topic, args.debug_topic]):
            bag_t = t.to_sec()
            if start_time is None:
                start_time = bag_t
            rel_t = bag_t - start_time
            if rel_t < args.warmup:
                continue
            if topic == args.truth_topic:
                truth_rows.append(pose_to_row(msg, bag_t))
            elif topic == args.pred_topic:
                pred_rows.append(pose_to_row(msg, bag_t))
            elif topic == args.debug_topic:
                try:
                    data = json.loads(msg.data)
                except Exception:
                    continue
                target = data.get("target") or {}
                detected = target.get("detected")
                confidence = target.get("confidence")
                max_objectness = data.get("max_objectness")
                debug_rows.append(
                    {
                        "t": bag_t,
                        "detected": detected,
                        "confidence": confidence,
                        "max_objectness": max_objectness,
                        "target_source": data.get("target_source"),
                        "selection_reason": (data.get("selection") or {}).get("reason"),
                    }
                )

    pairs = []
    truth_idx = 0
    for pred in pred_rows:
        truth, truth_idx = nearest_by_time(truth_rows, pred["t"], truth_idx)
        if truth is None:
            continue
        dt = pred["t"] - truth["t"]
        if abs(dt) > args.max_dt:
            continue
        truth_z = truth["z"] if args.truth_z_override is None else float(args.truth_z_override)
        dx = pred["x"] - truth["x"]
        dy = pred["y"] - truth["y"]
        dz = pred["z"] - truth_z
        pairs.append(
            {
                "pred_t": pred["t"],
                "truth_t": truth["t"],
                "dt": dt,
                "pred_x": pred["x"],
                "pred_y": pred["y"],
                "pred_z": pred["z"],
                "truth_x": truth["x"],
                "truth_y": truth["y"],
                "truth_z": truth_z,
                "err_x": dx,
                "err_y": dy,
                "err_z": dz,
                "err_xy": math.hypot(dx, dy),
                "err_3d": math.sqrt(dx * dx + dy * dy + dz * dz),
                "err_z_abs": abs(dz),
            }
        )

    pair_csv = os.path.join(args.out_dir, "target_regression_pairs.csv")
    if pairs:
        with open(pair_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(pairs[0].keys()))
            writer.writeheader()
            writer.writerows(pairs)

    debug_csv = os.path.join(args.out_dir, "target_detection_debug.csv")
    if debug_rows:
        with open(debug_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(debug_rows[0].keys()))
            writer.writeheader()
            writer.writerows(debug_rows)

    detected_values = [
        1.0 if row["detected"] else 0.0
        for row in debug_rows
        if row["detected"] is not None
    ]
    summary = {
        "bag": os.path.abspath(args.bag),
        "truth_topic": args.truth_topic,
        "pred_topic": args.pred_topic,
        "debug_topic": args.debug_topic,
        "max_dt": args.max_dt,
        "warmup": args.warmup,
        "truth_z_override": args.truth_z_override,
        "truth_samples": len(truth_rows),
        "pred_samples": len(pred_rows),
        "debug_samples": len(debug_rows),
        "paired_samples": len(pairs),
        "detected_ratio": (sum(detected_values) / len(detected_values)) if detected_values else None,
        "confidence": finite_stats(row["confidence"] for row in debug_rows if row["confidence"] is not None),
        "max_objectness": finite_stats(row["max_objectness"] for row in debug_rows if row["max_objectness"] is not None),
        "err_3d_m": finite_stats(row["err_3d"] for row in pairs),
        "err_xy_m": finite_stats(row["err_xy"] for row in pairs),
        "err_z_abs_m": finite_stats(row["err_z_abs"] for row in pairs),
        "dt_abs_s": finite_stats(abs(row["dt"]) for row in pairs),
        "pair_csv": pair_csv,
        "debug_csv": debug_csv,
    }
    summary_path = os.path.join(args.out_dir, "target_regression_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
