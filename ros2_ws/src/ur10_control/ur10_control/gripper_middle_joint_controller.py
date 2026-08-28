#!/usr/bin/env python3
"""Independent control for the Robotiq 3F middle finger.

The side fingers move together as a coordinated pair, while the middle finger keeps
its own command range and setpoint. This avoids coupling the middle finger into the
same motion profile as the two side fingers and avoids mirrored / flipped joint
orientation changes in the gripper model.
"""

from __future__ import annotations

from dataclasses import dataclass


class _JointController:
    def __init__(self, min_value: float, max_value: float) -> None:
        if min_value > max_value:
            raise ValueError("min_value must be <= max_value")
        self.min_value = min_value
        self.max_value = max_value

    def clamp(self, value: float) -> float:
        return min(max(float(value), self.min_value), self.max_value)


class SideFingerController(_JointController):
    """Keeps the two side fingers synchronized without coupling them to the middle joint."""

    def command(self, width: float) -> tuple[float, float]:
        clipped = self.clamp(width)
        return (clipped, clipped)


class MiddleJointController(_JointController):
    """Controls the middle finger independently and keeps the physical orientation fixed."""

    def command(self, position: float) -> float:
        # Invert the middle-joint command so the physical open/close direction matches
        # the user-facing gripper intent instead of the mirrored joint sign.
        mirrored = self.min_value + self.max_value - float(position)
        return self.clamp(mirrored)


@dataclass(frozen=True)
class GripperState:
    side_fingers: tuple[float, float]
    middle_joint: float


class Robotiq3FGripperController:
    """A separate controller for the middle finger while preserving the model's orientation."""

    def __init__(
        self,
        min_position: float = 0.0495,
        max_position: float = 1.2218,
    ) -> None:
        self.side_fingers = SideFingerController(min_position, max_position)
        self.middle_joint = MiddleJointController(min_position, max_position)

    def move(self, side_width: float, middle_position: float) -> GripperState:
        side_values = self.side_fingers.command(side_width)
        middle_value = self.middle_joint.command(middle_position)
        return GripperState(side_fingers=side_values, middle_joint=middle_value)

    def trajectory(self, side_width: float, middle_position: float) -> dict[str, float]:
        state = self.move(side_width, middle_position)
        return {
            "gripper_finger_1_joint_1": state.side_fingers[0],
            "gripper_finger_2_joint_1": state.side_fingers[1],
            "gripper_finger_middle_joint_1": state.middle_joint,
        }
