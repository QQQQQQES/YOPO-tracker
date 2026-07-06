import argparse
import json
import math

import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_world_topic", type=str, default="/yopo_runtime_test/yopov2/target_world")
    parser.add_argument("--tracking_debug_topic", type=str, default="/yopo_runtime_test/yopov2/tracking_debug")
    parser.add_argument("--rate", type=float, default=20.0)
    parser.add_argument("--x", type=float, default=6.0)
    parser.add_argument("--y", type=float, default=0.0)
    parser.add_argument("--z", type=float, default=2.0)
    parser.add_argument("--circle_radius", type=float, default=0.0)
    parser.add_argument("--circle_period", type=float, default=20.0)
    parser.add_argument("--confidence", type=float, default=0.95)
    return parser.parse_args()


def main():
    args = parse_args()
    rospy.init_node("yopov2_synthetic_target", anonymous=False)
    target_pub = rospy.Publisher(args.target_world_topic, PoseStamped, queue_size=5)
    debug_pub = rospy.Publisher(args.tracking_debug_topic, String, queue_size=5)
    rate = rospy.Rate(args.rate)
    start = rospy.Time.now().to_sec()
    rospy.loginfo("synthetic target publishing target=%s debug=%s", args.target_world_topic, args.tracking_debug_topic)

    while not rospy.is_shutdown():
        now = rospy.Time.now()
        t = now.to_sec() - start
        x = args.x
        y = args.y
        if args.circle_radius > 0.0:
            phase = 2.0 * math.pi * t / max(args.circle_period, 1e-3)
            x += args.circle_radius * math.cos(phase)
            y += args.circle_radius * math.sin(phase)

        pose = PoseStamped()
        pose.header.stamp = now
        pose.header.frame_id = "world"
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = args.z
        pose.pose.orientation.w = 1.0
        target_pub.publish(pose)

        debug = {
            "stamp": now.to_sec(),
            "detected": True,
            "confidence": args.confidence,
            "threshold": 0.5,
            "target_world": [x, y, args.z],
        }
        debug_pub.publish(json.dumps(debug, separators=(",", ":")))
        rate.sleep()


if __name__ == "__main__":
    main()
