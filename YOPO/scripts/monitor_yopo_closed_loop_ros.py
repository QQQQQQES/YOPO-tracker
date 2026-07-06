#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import time

import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String

try:
    from quadrotor_msgs.msg import PositionCommand
except ImportError:
    PositionCommand = None


def finite_mean(values):
    nums = []
    for value in values:
        if value is None:
            continue
        value = float(value)
        if math.isfinite(value):
            nums.append(value)
    return sum(nums) / len(nums) if nums else None


def counts(values):
    out = {}
    for value in values:
        out[str(value)] = out.get(str(value), 0) + 1
    return out


def dist3(a, b):
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--duration", type=float, default=25.0)
    parser.add_argument("--odom_topic", default="/yopo_runtime_test/sim/odom")
    parser.add_argument("--cmd_topic", default="/yopo_runtime_test/so3_control/pos_cmd")
    parser.add_argument("--debug_topic", default="/yopo_net/tracking_debug")
    parser.add_argument("--target_topic", default="/flightmare/yopo/target_world")
    parser.add_argument("--target_follow_z", type=float, default=2.0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    state = {
        "target": [2.0, 0.0, float(args.target_follow_z)],
        "odom_rows": [],
        "cmd_rows": [],
        "debug_rows": [],
    }
    start_wall = time.time()

    def odom_cb(msg):
        p = msg.pose.pose.position
        v = msg.twist.twist.linear
        target = list(state["target"])
        pos = [p.x, p.y, p.z]
        state["odom_rows"].append({
            "t": time.time() - start_wall,
            "x": p.x,
            "y": p.y,
            "z": p.z,
            "vx": v.x,
            "vy": v.y,
            "vz": v.z,
            "target_x": target[0],
            "target_y": target[1],
            "target_z": target[2],
            "dist3": dist3(pos, target),
            "dist_xy": math.hypot(p.x - target[0], p.y - target[1]),
        })

    def target_cb(msg):
        state["target"] = [
            msg.pose.position.x,
            msg.pose.position.y,
            float(args.target_follow_z),
        ]

    def debug_cb(msg):
        try:
            data = json.loads(msg.data)
        except Exception:
            return
        sel = data.get("selection") or {}
        target = data.get("target") or {}
        state["debug_rows"].append({
            "t": time.time() - start_wall,
            "selected_action": data.get("selected_action"),
            "selection_reason": sel.get("reason"),
            "target_source": data.get("target_source"),
            "min_score": data.get("min_score"),
            "selected_score": data.get("selected_score"),
            "max_objectness": data.get("max_objectness"),
            "selected_objectness": data.get("selected_objectness"),
            "selected_terminal_distance": sel.get("selected_terminal_distance"),
            "selected_terminal_progress": sel.get("selected_terminal_progress"),
            "selected_terminal_lateral_error": sel.get("selected_terminal_lateral_error"),
            "detected": target.get("detected"),
            "confidence": target.get("confidence"),
        })

    def cmd_cb(msg):
        state["cmd_rows"].append({
            "t": time.time() - start_wall,
            "x": msg.position.x,
            "y": msg.position.y,
            "z": msg.position.z,
            "vx": msg.velocity.x,
            "vy": msg.velocity.y,
            "vz": msg.velocity.z,
            "ax": msg.acceleration.x,
            "ay": msg.acceleration.y,
            "az": msg.acceleration.z,
            "yaw": msg.yaw,
            "yaw_dot": msg.yaw_dot,
        })

    rospy.init_node("monitor_yopo_closed_loop_ros", anonymous=True)
    rospy.Subscriber(args.odom_topic, Odometry, odom_cb, queue_size=100)
    rospy.Subscriber(args.target_topic, PoseStamped, target_cb, queue_size=10)
    rospy.Subscriber(args.debug_topic, String, debug_cb, queue_size=100)
    if PositionCommand is not None:
        rospy.Subscriber(args.cmd_topic, PositionCommand, cmd_cb, queue_size=200)

    end = time.time() + float(args.duration)
    while time.time() < end and not rospy.is_shutdown():
        time.sleep(0.05)

    odom_rows = state["odom_rows"]
    cmd_rows = state["cmd_rows"]
    debug_rows = state["debug_rows"]
    odom_csv = os.path.join(args.out_dir, "monitor_odom.csv")
    cmd_csv = os.path.join(args.out_dir, "monitor_cmd.csv")
    debug_csv = os.path.join(args.out_dir, "monitor_debug.csv")
    if odom_rows:
        with open(odom_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(odom_rows[0].keys()))
            writer.writeheader()
            writer.writerows(odom_rows)
    if cmd_rows:
        with open(cmd_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(cmd_rows[0].keys()))
            writer.writeheader()
            writer.writerows(cmd_rows)
    if debug_rows:
        with open(debug_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(debug_rows[0].keys()))
            writer.writeheader()
            writer.writerows(debug_rows)

    summary = {
        "duration_s": float(args.duration),
        "odom_samples": len(odom_rows),
        "cmd_samples": len(cmd_rows),
        "debug_samples": len(debug_rows),
        "odom_csv": odom_csv,
        "cmd_csv": cmd_csv,
        "debug_csv": debug_csv,
    }
    if odom_rows:
        d3 = [r["dist3"] for r in odom_rows]
        dxy = [r["dist_xy"] for r in odom_rows]
        summary.update({
            "start_pos": [odom_rows[0]["x"], odom_rows[0]["y"], odom_rows[0]["z"]],
            "end_pos": [odom_rows[-1]["x"], odom_rows[-1]["y"], odom_rows[-1]["z"]],
            "target": [odom_rows[-1]["target_x"], odom_rows[-1]["target_y"], odom_rows[-1]["target_z"]],
            "start_dist3": d3[0],
            "min_dist3": min(d3),
            "end_dist3": d3[-1],
            "start_dist_xy": dxy[0],
            "min_dist_xy": min(dxy),
            "end_dist_xy": dxy[-1],
            "max_z": max(r["z"] for r in odom_rows),
            "min_z": min(r["z"] for r in odom_rows),
        })
    if cmd_rows:
        summary.update({
            "start_cmd_pos": [cmd_rows[0]["x"], cmd_rows[0]["y"], cmd_rows[0]["z"]],
            "end_cmd_pos": [cmd_rows[-1]["x"], cmd_rows[-1]["y"], cmd_rows[-1]["z"]],
            "max_cmd_z": max(r["z"] for r in cmd_rows),
            "min_cmd_z": min(r["z"] for r in cmd_rows),
        })
    if debug_rows:
        summary.update({
            "target_source_counts": counts(r["target_source"] for r in debug_rows),
            "selection_reason_counts": counts(r["selection_reason"] for r in debug_rows),
            "selected_action_counts": counts(r["selected_action"] for r in debug_rows),
            "mean_max_objectness": finite_mean(r["max_objectness"] for r in debug_rows),
            "detected_ratio": finite_mean(
                1.0 if r["detected"] else 0.0
                for r in debug_rows
                if r["detected"] is not None
            ),
            "mean_selected_terminal_progress": finite_mean(r["selected_terminal_progress"] for r in debug_rows),
            "mean_selected_terminal_distance": finite_mean(r["selected_terminal_distance"] for r in debug_rows),
            "last_debug": debug_rows[-1],
        })

    summary_path = os.path.join(args.out_dir, "monitor_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
