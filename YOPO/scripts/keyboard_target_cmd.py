#!/usr/bin/env python3
import argparse
import select
import sys
import termios
import tty

import rospy
from geometry_msgs.msg import Twist


HELP = """
Keyboard target control

  w/s : +x / -x
  a/d : +y / -y
  r/f : +z / -z
  x or Space : stop
  +/- : speed up/down
  q : quit
"""


def read_key(timeout):
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    if not ready:
        return None
    return sys.stdin.read(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/flightmare/yopo/target_cmd_vel")
    parser.add_argument("--speed", type=float, default=0.8)
    parser.add_argument("--z_speed", type=float, default=0.4)
    parser.add_argument("--rate", type=float, default=20.0)
    args = parser.parse_args(rospy.myargv()[1:])

    rospy.init_node("keyboard_target_cmd", anonymous=True)
    pub = rospy.Publisher(args.topic, Twist, queue_size=1)
    rate = rospy.Rate(args.rate)

    old_settings = termios.tcgetattr(sys.stdin)
    speed = float(args.speed)
    z_speed = float(args.z_speed)
    cmd = Twist()

    print(HELP)
    print("Publishing Twist to {}".format(args.topic))
    print("Current speed: xy={:.2f} m/s, z={:.2f} m/s".format(speed, z_speed))

    try:
        tty.setcbreak(sys.stdin.fileno())
        while not rospy.is_shutdown():
            key = read_key(1.0 / max(args.rate, 1.0))
            if key:
                if key == "q":
                    break
                if key == "w":
                    cmd.linear.x = speed
                    cmd.linear.y = 0.0
                    cmd.linear.z = 0.0
                elif key == "s":
                    cmd.linear.x = -speed
                    cmd.linear.y = 0.0
                    cmd.linear.z = 0.0
                elif key == "a":
                    cmd.linear.x = 0.0
                    cmd.linear.y = speed
                    cmd.linear.z = 0.0
                elif key == "d":
                    cmd.linear.x = 0.0
                    cmd.linear.y = -speed
                    cmd.linear.z = 0.0
                elif key == "r":
                    cmd.linear.x = 0.0
                    cmd.linear.y = 0.0
                    cmd.linear.z = z_speed
                elif key == "f":
                    cmd.linear.x = 0.0
                    cmd.linear.y = 0.0
                    cmd.linear.z = -z_speed
                elif key in ("x", " "):
                    cmd = Twist()
                elif key in ("+", "="):
                    speed = min(speed + 0.1, 5.0)
                    z_speed = min(z_speed + 0.05, 2.0)
                    print("speed: xy={:.2f} m/s, z={:.2f} m/s".format(speed, z_speed))
                elif key in ("-", "_"):
                    speed = max(speed - 0.1, 0.1)
                    z_speed = max(z_speed - 0.05, 0.05)
                    print("speed: xy={:.2f} m/s, z={:.2f} m/s".format(speed, z_speed))
            pub.publish(cmd)
            rate.sleep()
    finally:
        pub.publish(Twist())
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


if __name__ == "__main__":
    main()
