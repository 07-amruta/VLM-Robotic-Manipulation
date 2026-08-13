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

## License

This project is licensed under the **MIT License**.
