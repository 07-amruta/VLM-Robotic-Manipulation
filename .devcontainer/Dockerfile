FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0

ENV DEBIAN_FRONTEND=noninteractive
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/cu12/lib:${LD_LIBRARY_PATH}

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
    opencv-python \
    tqdm

# OpenBLAS (required by Jetson PyTorch wheel at runtime)
RUN apt-get update && apt-get install -y \
    libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

# PyTorch (JetPack 6.2 / CUDA 12.6, cp310 aarch64)
RUN pip3 install --no-cache-dir \
    --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 \
    torch==2.11.0 torchvision==0.26.0

# cuDSS runtime (torch 2.11.0 on Jetson links against this; not bundled in the jp6 wheel)
RUN pip3 install --no-cache-dir nvidia-cudss-cu12

RUN pip3 install --no-cache-dir nvidia-cudss-cu12

# pip's CUDA 12.9 cuBLAS shadows JetPack's 12.6 cuBLAS and breaks cublasCreate on the Orin
RUN pip3 uninstall -y nvidia-cublas-cu12

# ninja for faster CUDA extension builds
RUN pip3 install --no-cache-dir ninja

# GraspNet-baseline: clone, patch for PyTorch 2.x compatibility, build extensions.
# inference (top-N grasps by score), not dataset loading/benchmark eval/NMS,
# which is all graspnetAPI provides. Revisit only if benchmark eval is needed later.
ENV TORCH_CUDA_ARCH_LIST="8.7"
RUN git clone https://github.com/graspnet/graspnet-baseline /opt/graspnet-baseline && \
    git -C /opt/graspnet-baseline checkout 280c215129f759ed8649cb4e89fc5dfee55f4f80 && \
    sed -i -E 's/\.data<([a-zA-Z_]+)>\(\)/\.data_ptr<\1>()/g' /opt/graspnet-baseline/pointnet2/_ext_src/src/*.cpp && \
    sed -i 's/from torch._six import container_abcs/import collections.abc as container_abcs/' /opt/graspnet-baseline/dataset/graspnet_dataset.py

RUN cd /opt/graspnet-baseline/pointnet2 && python3 setup.py install
RUN cd /opt/graspnet-baseline/knn && python3 setup.py install

# graspnetAPI — installed from source with heavy/unavailable deps (open3d, grasp_nms,
# autolab_core, etc.) stripped out. Only GraspGroup/Grasp classes needed for base-level
# use (sort, filter, transform grasp poses); dataset loading (GraspNet/GraspNetEval)
# and 3D visualization deliberately excluded — see grasp.py/__init__.py patches below.
RUN git config --global http.postBuffer 524288000 && \
    git config --global http.lowSpeedLimit 0 && \
    git config --global http.lowSpeedTime 999999 && \
    git clone https://github.com/graspnet/graspnetAPI /opt/graspnetAPI && \
    git -C /opt/graspnetAPI checkout eb57dd2092d8dbe05312a29c3d0c22f3226efbfc && \
    python3 -c "\
path = '/opt/graspnetAPI/graspnetAPI/__init__.py'; \
open(path, 'w').write('__author__ = \'mhgou\'\n__version__ = \'1.2.11\'\n\nfrom .grasp import Grasp, GraspGroup, RectGrasp, RectGraspGroup\n')" && \
    python3 -c "\
path = '/opt/graspnetAPI/graspnetAPI/grasp.py'; \
c = open(path).read(); \
c = c.replace('import open3d as o3d', 'try:\n    import open3d as o3d\nexcept ImportError:\n    o3d = None', 1); \
open(path, 'w').write(c)" && \
    python3 -c "\
path = '/opt/graspnetAPI/graspnetAPI/utils/utils.py'; \
c = open(path).read(); \
c = c.replace('import open3d as o3d', 'try:\n    import open3d as o3d\nexcept ImportError:\n    o3d = None', 1); \
open(path, 'w').write(c)"

RUN pip3 install --no-cache-dir transforms3d
RUN pip3 install --no-cache-dir --no-build-isolation --no-deps -e /opt/graspnetAPI

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

ENV GRASPNET_ROOT=/opt/graspnet-baseline