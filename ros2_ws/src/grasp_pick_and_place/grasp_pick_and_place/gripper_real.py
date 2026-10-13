import time

from grasp_pick_and_place.gripper import GripperMixin
from grasp_pick_and_place.robotiq_3f_gripper import RobotiqGripper3F

CONTACT_FIELDS = ('gDTA', 'gDTB', 'gDTC')  # 2 = stopped on contact while closing
STOPPED = (2, 3)                           # gSTA: all fingers stopped / at requested position


class RealGripperMixin(GripperMixin):
    """Same interface as GripperMixin over Modbus. The gripper stops itself on contact."""

    _hw = None

    def hardware(self):
        if self._hw is None:
            hw = RobotiqGripper3F(ip=self.get_parameter('gripper_ip').value,
                                  port=self.get_parameter('gripper_port').value)
            if not hw.activate():
                raise RuntimeError('Gripper activation failed')
            self._hw = hw
        return self._hw

    def move_gripper(self, positions):
        """Open only; joint targets have no meaning for this gripper."""
        ok = self.hardware().open(speed=self.get_parameter('gripper_speed').value)
        if not ok:
            self.get_logger().error('Gripper open failed')
        return ok

    def close_gripper(self):
        """Closes at the configured force; True only if the object is held."""
        hw = self.hardware()
        if not hw.close(force=self.get_parameter('gripper_force').value,
                        speed=self.get_parameter('gripper_speed').value):
            self.get_logger().error('Gripper close failed')
            return False

        deadline = time.monotonic() + self.get_parameter('gripper_settle_timeout').value
        status = hw.get_status()
        while status and status['gSTA'] not in STOPPED and time.monotonic() < deadline:
            time.sleep(0.05)
            status = hw.get_status()
        if not status or status['gSTA'] not in STOPPED:
            self.get_logger().error('Gripper fingers did not settle')
            return False
        if status['fault_status'] != 0:
            self.get_logger().error(f"Gripper fault 0x{status['fault_status']:02X}")
            return False

        contacts = sum(status[f] == 2 for f in CONTACT_FIELDS)
        self.get_logger().info(f'Gripper contacts: {contacts}/3')
        return contacts >= self.get_parameter('gripper_min_contacts').value