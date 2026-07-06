#!/usr/bin/env python3
import argparse
import csv
import json
import os
import time

import cv2
import numpy as np
import rospy
import torch
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Image

from config.config import cfg
from policy.yopo_network import YopoNetwork
from policy.state_transform import StateTransform
from policy.primitive import LatticePrimitive


def wait_for_message(topic, msg_type, timeout):
    rospy.loginfo("waiting for %s", topic)
    return rospy.wait_for_message(topic, msg_type, timeout=timeout)


def decode_rgb(msg, width, height):
    encoding = msg.encoding.lower()
    if encoding in ("rgb8", "bgr8"):
        image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3).copy()
        if encoding == "bgr8":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif encoding in ("mono8", "8uc1"):
        gray = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    else:
        raise ValueError(f"unsupported RGB encoding: {msg.encoding}")
    if image.shape[:2] != (height, width):
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return image.astype(np.float32) / 255.0


def decode_depth(msg, width, height, min_dis=0.04, max_dis=20.0):
    if msg.encoding == "32FC1":
        depth_m = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width).copy()
    elif msg.encoding == "16UC1":
        depth_m = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width).astype(np.float32) / 1000.0
    else:
        raise ValueError(f"unsupported depth encoding: {msg.encoding}")
    if depth_m.shape[:2] != (height, width):
        depth_m = cv2.resize(depth_m, (width, height), interpolation=cv2.INTER_NEAREST)
    depth = np.minimum(depth_m, max_dis) / max_dis
    nan_mask = np.isnan(depth) | (depth < min_dis / max_dis)
    image = cv2.inpaint(np.uint8(depth * 255), np.uint8(nan_mask), 1, cv2.INPAINT_NS)
    return depth_m, image.astype(np.float32) / 255.0


def target_uvd_to_network_frame(target_uvd, fx, fy, cx, cy, min_dis=0.04, max_dis=20.0):
    depth = float(target_uvd[2])
    if not np.isfinite(depth) or depth <= min_dis or depth > max_dis:
        return None
    right = ((float(target_uvd[0]) - float(cx)) / float(fx)) * depth
    down = ((float(target_uvd[1]) - float(cy)) / float(fy)) * depth
    out = np.array([depth, -right, -down], dtype=np.float64)
    return out if np.all(np.isfinite(out)) else None


def target_uvd_to_body_frame(target_uvd, fx, fy, cx, cy, Rcl=None, Pcl=None, min_dis=0.04, max_dis=20.0):
    depth = float(target_uvd[2])
    if not np.isfinite(depth) or depth <= min_dis or depth > max_dis:
        return None
    right = ((float(target_uvd[0]) - float(cx)) / float(fx)) * depth
    down = ((float(target_uvd[1]) - float(cy)) / float(fy)) * depth
    if Rcl is None:
        out = np.array([depth, -right, -down], dtype=np.float64)
    else:
        target_optical = np.array([right, down, depth], dtype=np.float64)
        out = Rcl.T.dot(target_optical - Pcl)
    return out if np.all(np.isfinite(out)) else None


def select_action(mode, score, objectness, threshold):
    if objectness is None or mode == "score":
        return int(np.argmin(score)), "score_only"
    if mode == "objectness":
        return int(np.argmax(objectness)), "max_objectness"
    if mode == "hybrid":
        best = int(np.argmax(objectness))
        if float(objectness[best]) >= threshold:
            return best, "hybrid_objectness"
        return int(np.argmin(score)), "hybrid_score_fallback"
    if mode == "paper":
        valid = objectness >= threshold
        if np.any(valid):
            ids = np.flatnonzero(valid)
            return int(ids[np.argmin(score[ids])]), "objectness_filter_min_score"
        return int(np.argmin(score)), "paper_score_fallback"
    raise ValueError(mode)


def draw_topdown(rows, out_path, selected_ids, target_xy):
    scale = 45.0
    margin = 80
    canvas = np.full((760, 760, 3), 245, dtype=np.uint8)
    origin = np.array([margin, canvas.shape[0] - margin], dtype=np.float32)

    def project(x, y):
        return int(origin[0] + x * scale), int(origin[1] - y * scale)

    for gx in range(0, 11):
        x0, y0 = project(gx, -8)
        x1, y1 = project(gx, 8)
        cv2.line(canvas, (x0, y0), (x1, y1), (225, 225, 225), 1)
    for gy in range(-8, 9):
        x0, y0 = project(0, gy)
        x1, y1 = project(10, gy)
        cv2.line(canvas, (x0, y0), (x1, y1), (225, 225, 225), 1)

    cv2.circle(canvas, project(0, 0), 7, (0, 0, 0), -1)
    cv2.putText(canvas, "odom", (project(0, 0)[0] + 8, project(0, 0)[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    if target_xy is not None:
        cv2.drawMarker(canvas, project(target_xy[0], target_xy[1]), (0, 160, 0),
                       markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
        cv2.putText(canvas, "gt/root", (project(target_xy[0], target_xy[1])[0] + 8,
                                        project(target_xy[0], target_xy[1])[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 130, 0), 1, cv2.LINE_AA)

    for r in rows:
        aid = int(r["action_id"])
        end = np.array([float(r["end_x_rel"]), float(r["end_y_rel"])])
        color = (150, 150, 150)
        thickness = 1
        if aid == selected_ids.get("paper"):
            color, thickness = (0, 215, 255), 3
        if aid == selected_ids.get("objectness"):
            color, thickness = (255, 200, 0), 3
        if aid == selected_ids.get("score"):
            color, thickness = (0, 0, 255), 2
        cv2.arrowedLine(canvas, project(0, 0), project(end[0], end[1]), color, thickness, tipLength=0.08)
        cv2.putText(canvas, str(aid), project(end[0], end[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    legend = "red=min_score  cyan=max_obj  yellow=paper"
    cv2.putText(canvas, legend, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.imwrite(out_path, canvas)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", required=True)
    parser.add_argument("--rgb_topic", default="/flightmare/yopo/rgb")
    parser.add_argument("--depth_topic", default="/flightmare/yopo/depth")
    parser.add_argument("--odom_topic", default="/yopo_runtime_test/sim/odom")
    parser.add_argument("--target_topic", default="/flightmare/yopo/target_world")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--selection_mode", default="paper", choices=["score", "objectness", "hybrid", "paper"])
    parser.add_argument("--objectness_threshold", type=float, default=0.5)
    parser.add_argument("--goal", type=float, nargs=3, default=None)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--pitch_angle_deg", type=float, default=0.0)
    parser.add_argument("--target_camera_Rcl", type=float, nargs=9, default=None)
    parser.add_argument("--target_camera_Pcl", type=float, nargs=3, default=None)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rospy.init_node("analyze_yopo_candidates_ros", anonymous=True)

    cfg["train"] = False
    height = int(cfg["image_height"])
    width = int(cfg["image_width"])
    image_channels = int(cfg.get("image_channels", 1))
    min_dis, max_dis = 0.04, 20.0
    fx = cfg.get("fx", None)
    fy = cfg.get("fy", None)
    cx = cfg.get("cx", width * 0.5)
    cy = cfg.get("cy", height * 0.5)
    if fx is None:
        fx = width / (2.0 * np.tan(np.deg2rad(cfg["horizon_camera_fov"]) * 0.5))
    if fy is None:
        fy = height / (2.0 * np.tan(np.deg2rad(cfg["vertical_camera_fov"]) * 0.5))
    target_camera_Rcl = np.asarray(args.target_camera_Rcl, dtype=np.float64).reshape(3, 3) if args.target_camera_Rcl is not None else None
    target_camera_Pcl = np.asarray(args.target_camera_Pcl, dtype=np.float64).reshape(3) if args.target_camera_Pcl is not None else None
    if (target_camera_Rcl is None) != (target_camera_Pcl is None):
        raise ValueError("--target_camera_Rcl and --target_camera_Pcl must be provided together")

    rgb_msg = wait_for_message(args.rgb_topic, Image, args.timeout)
    depth_msg = wait_for_message(args.depth_topic, Image, args.timeout)
    odom_msg = wait_for_message(args.odom_topic, Odometry, args.timeout)
    try:
        target_msg = wait_for_message(args.target_topic, PoseStamped, 1.0)
        target_world = np.array([target_msg.pose.position.x, target_msg.pose.position.y, target_msg.pose.position.z], dtype=np.float64)
    except Exception:
        target_world = None

    rgb = decode_rgb(rgb_msg, width, height)
    depth_m, depth_norm = decode_depth(depth_msg, width, height, min_dis=min_dis, max_dis=max_dis)
    if image_channels == 1:
        network_input = depth_norm.reshape(1, 1, height, width)
    elif image_channels == 3:
        network_input = np.transpose(rgb, (2, 0, 1)).reshape(1, 3, height, width)
    elif image_channels == 4:
        network_input = np.concatenate(
            (np.transpose(rgb, (2, 0, 1)).reshape(1, 3, height, width),
             depth_norm.reshape(1, 1, height, width)),
            axis=1,
        )
    else:
        raise ValueError(f"unsupported image_channels={image_channels}")

    odom_pos = np.array([odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y, odom_msg.pose.pose.position.z], dtype=np.float64)
    odom_vel = np.array([odom_msg.twist.twist.linear.x, odom_msg.twist.twist.linear.y, odom_msg.twist.twist.linear.z], dtype=np.float64)
    Rotation_wb = R.from_quat([
        odom_msg.pose.pose.orientation.x,
        odom_msg.pose.pose.orientation.y,
        odom_msg.pose.pose.orientation.z,
        odom_msg.pose.pose.orientation.w,
    ]).as_matrix()
    Rotation_bc = R.from_euler("ZYX", [0, args.pitch_angle_deg, 0], degrees=True).as_matrix()
    Rotation_wc = Rotation_wb.dot(Rotation_bc)
    Rotation_cw = Rotation_wc.T

    vel_c = Rotation_cw.dot(odom_vel)
    acc_c = np.zeros(3, dtype=np.float64)
    obs = np.concatenate((vel_c, acc_c), axis=0).astype(np.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    state_transform = StateTransform()
    lattice = LatticePrimitive.get_instance()
    policy = YopoNetwork()
    policy.load_state_dict(torch.load(args.weight, weights_only=True))
    policy = policy.to(device)
    policy.eval()

    with torch.inference_mode():
        depth_tensor = torch.from_numpy(network_input.astype(np.float32)).to(device)
        obs_norm = state_transform.normalize_obs(torch.from_numpy(obs[None, :]))
        obs_input = state_transform.prepare_input(obs_norm).to(device)
        output = policy(depth_tensor, obs_input)
        if len(output) != 4:
            raise RuntimeError("checkpoint did not return YOPOv2 outputs")
        endstate_pred, score_pred, objectness_logit, target_raw = output
        objectness = torch.sigmoid(objectness_logit).cpu().numpy().reshape(lattice.traj_num)
        score = score_pred.cpu().numpy().reshape(lattice.traj_num)
        endstate_raw = endstate_pred.cpu().numpy().reshape(9, lattice.traj_num).T
        target_uvd = state_transform.decode_target_cpu(target_raw.cpu().numpy().reshape(3, lattice.traj_num).T)

    lattice_ids = torch.arange(lattice.traj_num - 1, -1, -1)
    endstate_body = state_transform.pred_to_endstate_cpu(endstate_raw, lattice_ids)
    endstate_c = endstate_body.reshape(-1, 3, 3).transpose(0, 2, 1)
    endstate_w = np.matmul(Rotation_wc, endstate_c)

    selected_ids = {
        "score": select_action("score", score, objectness, args.objectness_threshold)[0],
        "objectness": select_action("objectness", score, objectness, args.objectness_threshold)[0],
        "paper": select_action("paper", score, objectness, args.objectness_threshold)[0],
    }
    requested_id, requested_reason = select_action(args.selection_mode, score, objectness, args.objectness_threshold)

    rows = []
    for aid in range(lattice.traj_num):
        end_rel = endstate_w[aid, :, 0]
        end_world = odom_pos + end_rel
        target_vec_c = target_uvd_to_body_frame(
            target_uvd[aid],
            fx,
            fy,
            cx,
            cy,
            Rcl=target_camera_Rcl,
            Pcl=target_camera_Pcl,
            min_dis=min_dis,
            max_dis=max_dis,
        )
        pred_world = odom_pos + Rotation_wc.dot(target_vec_c) if target_vec_c is not None else np.array([np.nan, np.nan, np.nan])
        true_dist = float(np.linalg.norm(end_world - target_world)) if target_world is not None else np.nan
        pred_dist = float(np.linalg.norm(end_world - pred_world)) if np.all(np.isfinite(pred_world)) else np.nan
        target_bearing = np.arctan2(pred_world[1] - odom_pos[1], pred_world[0] - odom_pos[0]) if np.all(np.isfinite(pred_world[:2])) else np.nan
        end_bearing = np.arctan2(end_rel[1], end_rel[0])
        rows.append({
            "action_id": aid,
            "score": float(score[aid]),
            "objectness": float(objectness[aid]),
            "target_u": float(target_uvd[aid, 0]),
            "target_v": float(target_uvd[aid, 1]),
            "target_depth": float(target_uvd[aid, 2]),
            "end_x_rel": float(end_rel[0]),
            "end_y_rel": float(end_rel[1]),
            "end_z_rel": float(end_rel[2]),
            "end_x_world": float(end_world[0]),
            "end_y_world": float(end_world[1]),
            "end_z_world": float(end_world[2]),
            "pred_target_x_world": float(pred_world[0]),
            "pred_target_y_world": float(pred_world[1]),
            "pred_target_z_world": float(pred_world[2]),
            "dist_end_to_true_root": true_dist,
            "dist_end_to_pred_target": pred_dist,
            "bearing_end_deg": float(np.degrees(end_bearing)),
            "bearing_pred_target_deg": float(np.degrees(target_bearing)) if np.isfinite(target_bearing) else np.nan,
            "is_min_score": aid == selected_ids["score"],
            "is_max_objectness": aid == selected_ids["objectness"],
            "is_paper": aid == selected_ids["paper"],
            "is_requested": aid == requested_id,
        })

    csv_path = os.path.join(args.out_dir, "candidate_actions.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    rgb_bgr = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    overlay = rgb_bgr.copy()
    for r0 in rows:
        aid = int(r0["action_id"])
        u = int(round(float(r0["target_u"])))
        v = int(round(float(r0["target_v"])))
        color = (140, 140, 140)
        if aid == selected_ids["score"]:
            color = (0, 0, 255)
        if aid == selected_ids["objectness"]:
            color = (255, 200, 0)
        if aid == selected_ids["paper"]:
            color = (0, 215, 255)
        cv2.drawMarker(overlay, (u, v), color, cv2.MARKER_CROSS, 7, 1)
        cv2.putText(overlay, str(aid), (u + 2, v + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.28, color, 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(args.out_dir, "candidate_image_overlay_x4.png"),
                cv2.resize(overlay, (width * 4, height * 4), interpolation=cv2.INTER_NEAREST))

    target_xy_rel = None
    if target_world is not None:
        target_xy_rel = (target_world - odom_pos)[:2]
    draw_topdown(rows, os.path.join(args.out_dir, "candidate_topdown.png"), selected_ids, target_xy_rel)

    summary = {
        "csv": csv_path,
        "selected": selected_ids,
        "requested_selection_mode": args.selection_mode,
        "requested_action": requested_id,
        "requested_reason": requested_reason,
        "max_objectness": float(np.max(objectness)),
        "min_score": float(np.min(score)),
        "valid_objectness_count": int(np.sum(objectness >= args.objectness_threshold)),
        "odom_world": odom_pos.tolist(),
        "goal_world": goal_world.tolist(),
        "target_world": target_world.tolist() if target_world is not None else None,
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
