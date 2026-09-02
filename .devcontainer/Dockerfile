FROM ros:humble-ros-base

ENV DEBIAN_FRONTEND=noninteractive

# ---- Core build tools ----
RUN apt-get update && apt-get install -y \
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

# ---- ROS 2 + xacro + joint state tools ----
RUN apt-get update && apt-get install -y \
    ros-humble-xacro \
    ros-humble-joint-state-publisher-gui \
    ros-humble-robot-state-publisher \
    && rm -rf /var/lib/apt/lists/*

# ---- ros2_control ----
RUN apt-get update && apt-get install -y \
    ros-humble-ros2-control \
    ros-humble-ros2-controllers \
    ros-humble-controller-manager \
    && rm -rf /var/lib/apt/lists/*

# ---- MoveIt 2 ----
RUN apt-get update && apt-get install -y \
    ros-humble-moveit \
    ros-humble-moveit-planners-ompl \
    ros-humble-moveit-ros-visualization \
    ros-humble-moveit-ros-move-group \
    ros-humble-moveit-simple-controller-manager \
    && rm -rf /var/lib/apt/lists/*

# ---- Ignition Gazebo Fortress ----
RUN apt-get update && apt-get install -y \
    ros-humble-ros-ign-gazebo \
    ros-humble-ros-ign-bridge \
    ros-humble-ign-ros2-control \
    && rm -rf /var/lib/apt/lists/*

# ---- RViz ----
RUN apt-get update && apt-get install -y \
    ros-humble-rviz2 \
    ros-humble-rviz-common \
    ros-humble-rviz-default-plugins \
    mesa-utils \
    && rm -rf /var/lib/apt/lists/*

# ---- Perception ----
RUN apt-get update && apt-get install -y \
    ros-humble-cv-bridge \
    ros-humble-vision-opencv \
    ros-humble-image-transport \
    ros-humble-perception-pcl \
    ros-humble-pcl-ros \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# ---- Python libraries ----
RUN pip3 install --no-cache-dir \
    numpy \
    scipy \
    matplotlib \
    opencv-python

# ---- rosdep init ----
RUN rosdep init || true
RUN rosdep update

# ---- Source ROS 2 setup on every shell ----
RUN echo "source /opt/ros/humble/setup.bash" >> /etc/bash.bashrc
RUN echo 'if [ -f /workspaces/VLM-Robotic-Manipulation/ros2_ws/install/setup.bash ]; then source /workspaces/VLM-Robotic-Manipulation/ros2_ws/install/setup.bash; fi' >> /etc/bash.bashrc

RUN mkdir -p /root/.bash_history_dir && \
    echo 'export HISTFILE=/root/.bash_history_dir/.bash_history' >> /etc/bash.bashrc && \
    echo 'export PROMPT_COMMAND="history -a; $PROMPT_COMMAND"' >> /etc/bash.bashrc

WORKDIR /workspaces