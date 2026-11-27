#!/usr/bin/env bash
# ==============================================================================
# Auto-installer for Doosan M0609 NMPC + SVM + D435 Vision + Gazebo Harmonic
# Supported OS: Ubuntu 24.04 (ROS 2 Jazzy)
# Usage:
#   chmod +x setup_environment.sh && ./setup_environment.sh
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROS_DISTRO="${ROS_DISTRO:-jazzy}"

# Auto-detect workspace root (<ws>/src/<pkg>), or create ~/ros2_ws if cloned standalone
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
if [ "$(basename "$PARENT_DIR")" = "src" ]; then
    WS_DIR="$(dirname "$PARENT_DIR")"
    SRC_DIR="$PARENT_DIR"
else
    WS_DIR="${WS_DIR:-$HOME/ros2_ws}"
    SRC_DIR="$WS_DIR/src"
    mkdir -p "$SRC_DIR"
    if [ ! -e "$SRC_DIR/dsr_m0609_nmpc" ]; then
        echo "[INFO] Linking package into workspace: $SRC_DIR/dsr_m0609_nmpc"
        ln -s "$SCRIPT_DIR" "$SRC_DIR/dsr_m0609_nmpc"
    fi
fi

echo "=================================================================="
echo " Workspace : $WS_DIR"
echo " Package   : $SCRIPT_DIR"
echo " ROS 2     : $ROS_DISTRO"
echo "=================================================================="

# 1. Install ROS 2 Jazzy repo if missing
sudo apt-get update
sudo apt-get install -y curl gnupg lsb-release software-properties-common build-essential cmake git python3-pip

if [ ! -f "/opt/ros/$ROS_DISTRO/setup.bash" ]; then
    echo "[STEP 1/5] Installing ROS 2 $ROS_DISTRO apt repository & desktop..."
    sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo "$UBUNTU_CODENAME") main" \
        | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
    sudo apt-get update
    sudo apt-get install -y "ros-${ROS_DISTRO}-desktop" python3-colcon-common-extensions python3-rosdep
else
    echo "[STEP 1/5] ROS 2 $ROS_DISTRO already installed."
    sudo apt-get install -y python3-colcon-common-extensions python3-rosdep
fi

# 2. Install Gazebo Harmonic, ros2_control, Vision, and GUI dependencies
echo "[STEP 2/5] Installing ROS 2 Gazebo, Control, OpenCV, and PyQt5 packages..."
sudo apt-get install -y \
    "ros-${ROS_DISTRO}-ros-gz" \
    "ros-${ROS_DISTRO}-ros-gz-sim" \
    "ros-${ROS_DISTRO}-ros-gz-bridge" \
    "ros-${ROS_DISTRO}-ros-gz-image" \
    "ros-${ROS_DISTRO}-gz-ros2-control" \
    "ros-${ROS_DISTRO}-ros2-control" \
    "ros-${ROS_DISTRO}-ros2-controllers" \
    "ros-${ROS_DISTRO}-controller-manager" \
    "ros-${ROS_DISTRO}-joint-state-broadcaster" \
    "ros-${ROS_DISTRO}-forward-command-controller" \
    "ros-${ROS_DISTRO}-robot-state-publisher" \
    "ros-${ROS_DISTRO}-tf2-ros" \
    "ros-${ROS_DISTRO}-tf2-ros-py" \
    "ros-${ROS_DISTRO}-cv-bridge" \
    "ros-${ROS_DISTRO}-xacro" \
    python3-pyqt5 \
    python3-pyqt5.qtsvg \
    python3-opencv \
    python3-numpy

# 3. Install & build acados if missing
ACADOS_DIR="${ACADOS_SOURCE_DIR:-${ACADOS_DIR:-$HOME/acados}}"
if [ ! -f "$ACADOS_DIR/lib/libacados.so" ] && [ -f "/opt/acados/lib/libacados.so" ]; then
    ACADOS_DIR="/opt/acados"
fi

if [ ! -f "$ACADOS_DIR/lib/libacados.so" ]; then
    echo "[STEP 3/5] Installing acados into $ACADOS_DIR ..."
    git clone https://github.com/acados/acados.git "$ACADOS_DIR"
    (
        cd "$ACADOS_DIR"
        git submodule update --recursive --init
        mkdir -p build && cd build
        cmake -DCMAKE_INSTALL_PREFIX="$ACADOS_DIR" ..
        make install -j"$(nproc)"
    )
else
    echo "[STEP 3/5] Found acados at $ACADOS_DIR"
fi

export ACADOS_SOURCE_DIR="$ACADOS_DIR"
export LD_LIBRARY_PATH="$ACADOS_DIR/lib:${LD_LIBRARY_PATH:-}"

# 4. Clone doosan-robot2 (provides dsr_msgs2 + dsr_description2 M0609 meshes)
if [ ! -d "$SRC_DIR/doosan-robot2" ]; then
    echo "[STEP 4/5] Cloning doosan-robot2 into $SRC_DIR/doosan-robot2 ..."
    git clone https://github.com/doosan-robotics/doosan-robot2.git "$SRC_DIR/doosan-robot2"
else
    echo "[STEP 4/5] Found doosan-robot2 at $SRC_DIR/doosan-robot2"
fi

# Initialize rosdep if needed & install workspace package deps
if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then
    sudo rosdep init || true
fi
rosdep update || true

set +u
source "/opt/ros/$ROS_DISTRO/setup.bash"
set -u
rosdep install --from-paths "$SRC_DIR" --ignore-src -r -y || true

# 5. Recompile solver shared lib natively for this CPU & build ROS 2 packages
echo "[STEP 5/5] Rebuilding NMPC solver .so & compiling ROS 2 workspace..."
make -C "$SCRIPT_DIR/c_generated_solver_kinematic" clean ocp_shared_lib

cd "$WS_DIR"
colcon build --packages-up-to dsr_m0609_nmpc --symlink-install

echo ""
echo "=================================================================="
echo " SETUP COMPLETED SUCCESSFULLY!"
echo " Run GUI with:"
echo "   cd $WS_DIR"
echo "   source /opt/ros/$ROS_DISTRO/setup.bash"
echo "   source install/setup.bash"
echo "   ros2 launch dsr_m0609_nmpc teach_pendant.launch.py"
echo "=================================================================="
