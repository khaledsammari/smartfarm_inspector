#!/usr/bin/env bash
# Install smartfarm_inspector into an existing Eagle robotics workspace.
#
#   ./install.sh [workspace]      default: ~/eagle_robotics_ws
#
# Idempotent: safe to re-run. Symlinks this directory into the workspace so
# you edit in one place and colcon builds from the same files.
set -euo pipefail

WS="${1:-$HOME/eagle_robotics_ws}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="smartfarm_inspector"

echo "==> package : $HERE"
echo "==> workspace: $WS"

if [ ! -d "$WS/src" ]; then
  echo "ERROR: $WS/src does not exist."
  echo "Point this at your Eagle workspace: ./install.sh /path/to/ws"
  exit 1
fi

# --- 1. Python deps -------------------------------------------------------
echo "==> installing Python dependencies"
python3 -m pip install --user -q pyyaml pytest || {
  echo "pip failed; try: python3 -m pip install --break-system-packages pyyaml pytest"
}

# --- 2. self-test before touching the workspace ---------------------------
echo "==> running tests"
cd "$HERE"
python3 -m pytest tests -q || { echo "TESTS FAILED - not installing"; exit 1; }

# --- 3. link into the workspace ------------------------------------------
if [ -e "$WS/src/$PKG" ] && [ ! -L "$WS/src/$PKG" ]; then
  echo "WARNING: $WS/src/$PKG exists and is a real directory, not a symlink."
  echo "Remove or rename it first, then re-run."
  exit 1
fi
ln -sfn "$HERE" "$WS/src/$PKG"
echo "==> linked $WS/src/$PKG -> $HERE"

# --- 4. verify the field map against the sim map -------------------------
MAP="$WS/src/robot_navigation/nav_launch/maps/car_tree.yaml"
if [ -f "$MAP" ]; then
  echo "==> verifying field map against $MAP"
  python3 -m pip install --user -q pillow scipy numpy 2>/dev/null || true
  python3 -m smartfarm.tools.verify_field_map --map "$MAP" | tail -3 || \
    echo "   (verifier needs pillow/scipy/numpy; skipping)"
else
  echo "==> skipping map check, $MAP not found"
fi

# --- 5. build -------------------------------------------------------------
echo "==> building"
cd "$WS"
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
colcon build --packages-select "$PKG" --symlink-install

cat <<'DONE'

==> done.

Next:
  source ~/eagle_robotics_ws/install/setup.bash

  # terminal 1
  ros2 launch sim_bringup go2_sim.launch.py

  # terminal 2
  ros2 launch smartfarm_inspector smartfarm.launch.py auto_launch:=false

  # terminal 3
  ros2 topic pub --once /farm/inspection_request std_msgs/msg/String \
    "data: 'check the north field for powdery mildew on tomatoes'"
  ros2 topic echo /farm/inspection_result

Claude on your subscription (no API credits):
  npm install -g @anthropic-ai/claude-code && claude login
  pip install claude-agent-sdk
  unset ANTHROPIC_API_KEY
  python3 -m smartfarm.tools.check_auth
DONE
