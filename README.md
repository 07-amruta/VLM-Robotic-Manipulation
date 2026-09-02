# VLM-Based Robotic Manipulation

A robotics project focused on **Vision-Language Model (VLM) based task planning for autonomous robotic manipulation** using ROS 2, MoveIt 2, RGB-D perception, and robotic manipulation.

The project begins with a **UR10 + Robotiq gripper** and will progressively extend to **dual-arm manipulation on a humanoid upper-body platform**.

## Pipeline

```text
Natural Language Command
          ↓
         VLM
          ↓
    Task Planning
          ↓
   Action Generation
          ↓
 ┌────────┴────────┐
 ↓                 ↓
Perception      Robot State
 └────────┬────────┘
          ↓
       MoveIt 2
          ↓
   Motion Planning
          ↓
   Robot Execution
          ↓
 Feedback / Verification
```

## Technologies

* **Robot:** Universal Robots UR10 + Robotiq Gripper
* **Perception:** Intel RealSense RGB-D
* **Middleware:** ROS 2
* **Motion Planning:** MoveIt 2
* **Simulation:** Gazebo / RViz
* **VLM:** Local or API-based
* **Edge Platform:** NVIDIA Jetson AGX Orin 64GB
* **Final Platform:** Dual-arm humanoid upper body

## Development Roadmap

```text
UR10 Simulation
      ↓
ROS 2 Control
      ↓
MoveIt 2
      ↓
Pick & Place
      ↓
RGB-D Perception
      ↓
Grasp Planning
      ↓
VLM Task Planning
      ↓
Jetson Deployment
      ↓
Dual-Arm Manipulation
```

## Example

The system should eventually understand commands such as:

> "Place all the red objects in the basket."

and execute:

```text
Find Objects
     ↓
Select Object
     ↓
Pick
     ↓
Place
     ↓
Verify
     ↓
Repeat
```

## Repository Structure

```text
vlm-robotic-manipulation/
├── src/
├── config/
├── launch/
├── scripts/
├── docs/
├── experiments/
├── README.md
├── LICENSE
└── .gitignore
```

## Current Status

**Stage: Initial Development**

Currently focusing on:

* UR10 simulation
* ROS 2
* MoveIt 2
* Robot control
* Basic manipulation

Perception, VLM, Jetson deployment, and dual-arm manipulation will be integrated progressively.

## ROS2 commands

**Simulation of UR10**

* All Packages - Robot Bringup Build

```bash
colcon build --packages-up-to robot_bringup --symlink-install --allow-overriding ur_description
```

* Packages Build Command

```bash
colcon build --packages-select robotiq_3f_gripper_description ur10_control ur10_gripper_moveit_config --symlink-install
```

* For starting UR10 Control Stack

```bash
ros2 launch ur_robot_driver ur10.launch.py robot_ip:=127.0.0.1 use_fake_hardware:=true
```

* For starting UR10 MOVEIt2 Config

```bash
ros2 launch ur_moveit_config ur_moveit.launch.py ur_type:=ur10 use_sim_time:=true
```

* For Gazebo Ignition Sim

```bash
ros2 launch ur10_control gazebo_simulation.launch.py
```

* For Gripper Moveit2 Sim

```bash
ros2 launch ur10_gripper_moveit_config move_group.launch.py
```

* For Hardware UR10 Arm Control

```bash
ros2 launch ur_robot_driver ur_control.launch.py   ur_type:=ur10   robot_ip:=192.168.1.102   kinematics_params_file:="${HOME}/my_robot_calibration.yaml"
```

* For Robot Arm Caliberation

```bash
ros2 launch ur_calibration calibration_correction.launch.py robot_ip:=192.168.1.102 target_filename:="${HOME}/my_robot_calibration.yaml"
```

* For Sourcing the Workspace

```bash
source install/setup.bash
```

## License

This project is licensed under the **MIT License**.
