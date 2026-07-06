#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  CONFIG="/mnt/nvme0n1p5/YOPO/YOPO/config/yopo_runtime_flightmare_image.yaml"
else
  CONFIG="$1"
  shift
fi

source /opt/ros/noetic/setup.bash
source /mnt/nvme0n1p5/YOPO/Controller/devel/setup.bash

cd /mnt/nvme0n1p5/YOPO/YOPO
exec /home/zml/anaconda3/envs/yopo/bin/python -u test_yopo_ros.py --runtime_config "$CONFIG" "$@"
