FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0

ENV DEBIAN_FRONTEND=noninteractive

# ROS 2 repository
RUN apt-get update && apt-get install -y \
    curl \
    gnupg2 \
    lsb-release \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg

RUN echo "deb [arch=arm64 signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu jammy main" \
    > /etc/apt/sources.list.d/ros2.list

# Core build tools and ROS 2
RUN apt-get update && apt-get install -y \
    ros-humble-ros-base \
    build-essential \
    cmake \
    g++ \
    git \
    wget \
    curl \
    nano \
    python3-pip \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
    && rm -rf /var/lib/apt/lists/*

# ROS 2 tools
RUN apt-get update && apt-get install -y \
    ros-humble-xacro \
    ros-humble-joint-state-publisher-gui \
    ros-humble-robot-state-publisher \
    && rm -rf /var/lib/apt/lists/*

# ros2_control
RUN apt-get update && apt-get install -y \
    ros-humble-ros2-control \
    ros-humble-ros2-controllers \
    ros-humble-controller-manager \
    && rm -rf /var/lib/apt/lists/*

# MoveIt 2
RUN apt-get update && apt-get install -y \
    ros-humble-moveit \
    ros-humble-moveit-planners-ompl \
    ros-humble-moveit-ros-visualization \
    ros-humble-moveit-ros-move-group \
    ros-humble-moveit-simple-controller-manager \
    && rm -rf /var/lib/apt/lists/*

# Gazebo Fortress - built from source.
# Not available as arm64 apt packages for Humble; official binaries only ship
# for amd64. Built into the image (not /workspaces) so it survives workspace
# rebuilds and doesn't depend on manual steps that get lost.
RUN apt-get update && apt-get install -y \
    python3-vcstool \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/fortress_ws/src && \
    cd /opt/fortress_ws/src && \
    wget https://raw.githubusercontent.com/gazebo-tooling/gazebodistro/master/collection-fortress.yaml && \
    vcs import < collection-fortress.yaml

RUN cd /opt/fortress_ws && \
    apt-get update && \
    apt-get install -y $(sort -u $(find . -iname 'packages-'`lsb_release -cs`'.apt' -o -iname 'packages.apt' | grep -v '/\.git/') | sed '/gz\|ignition/d' | tr '\n' ' ') \
    && rm -rf /var/lib/apt/lists/*

RUN cd /opt/fortress_ws && \
    colcon build --merge-install --cmake-args -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON --parallel-workers 12

RUN echo "source /opt/fortress_ws/install/setup.bash" >> /etc/bash.bashrc

# RViz
RUN apt-get update && apt-get install -y \
    ros-humble-rviz2 \
    ros-humble-rviz-common \
    ros-humble-rviz-default-plugins \
    mesa-utils \
    && rm -rf /var/lib/apt/lists/*

# Perception
RUN apt-get update && apt-get install -y \
    ros-humble-cv-bridge \
    ros-humble-vision-opencv \
    ros-humble-image-transport \
    ros-humble-perception-pcl \
    ros-humble-pcl-ros \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Python libraries
RUN pip3 install --no-cache-dir \
    numpy \
    scipy \
    matplotlib \
    opencv-python

# rosdep
RUN rosdep init || true
RUN rosdep update

# Source ROS 2 automatically
RUN echo "source /opt/ros/humble/setup.bash" >> /etc/bash.bashrc

RUN echo 'if [ -f /workspaces/VLM-Robotic-Manipulation/ros2_ws/install/setup.bash ]; then source /workspaces/VLM-Robotic-Manipulation/ros2_ws/install/setup.bash; fi' \
    >> /etc/bash.bashrc

# Persistent bash history
RUN mkdir -p /root/.bash_history_dir && \
    echo 'export HISTFILE=/root/.bash_history_dir/.bash_history' >> /etc/bash.bashrc && \
    echo 'export PROMPT_COMMAND="history -a; $PROMPT_COMMAND"' >> /etc/bash.bashrc

# Default working directory
WORKDIR /workspaces