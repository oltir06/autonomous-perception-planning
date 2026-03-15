#!/bin/bash
set -e

# Source ROS2 Humble setup
source /opt/ros/humble/setup.bash

# Source workspace overlay if it exists
if [ -f /workspace/install/setup.bash ]; then
    source /workspace/install/setup.bash
fi

exec "$@"
