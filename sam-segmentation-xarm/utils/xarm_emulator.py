"""
xArm API Simulator - Emulator of the XArmAPI

Usage:
    from xarm_simulator import XArmSimulator as XArmAPI
    
    robot = XArmAPI("192.168.1.231")
    robot.motion_enable(enable=True)
    robot.set_mode(0)
    robot.set_state(state=0)
"""

import time
import math
import threading
import random
from typing import Tuple, List, Optional


class XArmEmulator:
    CODE_SUCCESS = 0
    CODE_WARN = 1
    CODE_ERROR = -1

    STATE_READY = 0
    STATE_MOVING = 1
    STATE_PAUSED = 3
    STATE_ERROR = 4

    def __init__(self, ip: str = "127.0.0.1", is_radian: bool = False):
        self.ip = ip
        self.is_radian = is_radian

        self._connected = True
        self._motion_enabled = False
        self._mode = 0  # 0=position mode, 1=servo mode, etc.
        self._state = self.STATE_READY
        self._error_code = 0

        self._current_position = [200.0, 0.0, 200.0, 180.0, 0.0, 0.0]

        self._target_position = self._current_position.copy()

        # Workspace limits (mm)
        self._workspace_limits = {
            'x': (-700, 700),
            'y': (-700, 700),
            'z': (0, 700),
            'roll': (-360, 360),
            'pitch': (-360, 360),
            'yaw': (-360, 360)
        }

        self._max_speed = 1000
        self._current_speed = 100

        self._movement_thread = None
        self._stop_movement = False
        self._movement_lock = threading.Lock()

        self._total_distance = 0.0
        self._total_movements = 0

        print(f"XArmAPI Emulator Running...")

    def _format_position(self, pos: List[float]) -> str:
        return f"X:{pos[0]:.1f} Y:{pos[1]:.1f} Z:{pos[2]:.1f} R:{pos[3]:.1f} P:{pos[4]:.1f} Y:{pos[5]:.1f}"

    def connect(self) -> int:
        self._connected = True
        print("XArmAPI Emulator: connected.")
        return self.CODE_SUCCESS

    def disconnect(self) -> int:
        self._stop_movement = True
        if self._movement_thread and self._movement_thread.is_alive():
            self._movement_thread.join()
        self._connected = False
        print("XArmAPI Emulator: connection closed.")
        return self.CODE_SUCCESS

    def motion_enable(self, enable: bool = True, servo_id: Optional[int] = None) -> int:
        self._motion_enabled = enable
        status = "active" if enable else "stopped"
        print(f"XArmAPI Emulator: motors {status}")
        return self.CODE_SUCCESS

    def set_mode(self, mode: int) -> int:
        """
        Robot mode

        Args:
            mode: 0=position mode, 1=servo mode, 2=joint teaching mode
        """
        self._mode = mode
        mode_names = {0: "Position", 1: "Servo", 2: "Joint Teaching"}
        print(
            f"XArmAPI Emulator: mode {mode_names.get(mode, 'Unknown')} (mode={mode})")
        return self.CODE_SUCCESS

    def set_state(self, state: int) -> int:
        """
        Robot state

        Args:
            state: 0=ready, 3=pause, 4=stop
        """
        self._state = state
        state_names = {0: "Ready", 3: "Pause", 4: "Stop"}
        print(
            f"XArmAPI Emulator: state {state_names.get(state, 'Unknown')} (state={state})")
        return self.CODE_SUCCESS

    def get_state(self) -> Tuple[int, int]:
        """
        Get robot state

        Returns:
            (code, state)
        """
        return (self.CODE_SUCCESS, self._state)

    def get_position(self, is_radian: Optional[bool] = None) -> Tuple[int, List[float]]:
        """
        Get robot position

        Returns:
            (code, position)
        """
        with self._movement_lock:
            return (self.CODE_SUCCESS, self._current_position.copy())

    def get_servo_angle(self, servo_id: Optional[int] = None, is_radian: Optional[bool] = None) -> Tuple[int, List[float]]:
        """Get servo angles"""
        angles = [0.0, -45.0, 0.0, 0.0, 45.0, 0.0]
        return (self.CODE_SUCCESS, angles)

    def set_position(self, x: Optional[float] = None, y: Optional[float] = None,
                     z: Optional[float] = None, roll: Optional[float] = None,
                     pitch: Optional[float] = None, yaw: Optional[float] = None,
                     speed: Optional[float] = None, mvacc: Optional[float] = None,
                     mvtime: Optional[float] = None, relative: bool = False,
                     wait: bool = False, timeout: Optional[float] = None,
                     **kwargs) -> int:
        """
        Move to position

        Args:
            x, y, z: [mm]
            roll, pitch, yaw: [degrees]
            speed: [mm/s]
            wait: wait for movement
            relative: relative position
        """
        if not self._motion_enabled:
            print("XArmAPI Emulator: error motors are disabled.")
            return self.CODE_ERROR

        target = self._current_position.copy()

        if x is not None:
            target[0] = (target[0] + x) if relative else x
        if y is not None:
            target[1] = (target[1] + y) if relative else y
        if z is not None:
            target[2] = (target[2] + z) if relative else z
        if roll is not None:
            target[3] = (target[3] + roll) if relative else roll
        if pitch is not None:
            target[4] = (target[4] + pitch) if relative else pitch
        if yaw is not None:
            target[5] = (target[5] + yaw) if relative else yaw

        # check limits
        if not self._check_limits(target):
            print("XArmAPI Emulator: error workspace limits.")
            return self.CODE_ERROR

        if speed is not None:
            self._current_speed = min(speed, self._max_speed)

        self._target_position = target

        distance = self._calculate_distance(self._current_position, target)

        print(f"XArmAPI Emulator: move to {self._format_position(target)}")
        print(
            f"XArmAPI Emulator: distance {distance:.1f} mm, speed: {self._current_speed:.0f} mm/s")

        self._start_movement(wait)

        return self.CODE_SUCCESS

    def _check_limits(self, position: List[float]) -> bool:
        coords = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']
        for i, coord in enumerate(coords):
            min_val, max_val = self._workspace_limits[coord]
            if not (min_val <= position[i] <= max_val):
                return False
        return True

    def _calculate_distance(self, pos1: List[float], pos2: List[float]) -> float:
        return math.sqrt(
            (pos2[0] - pos1[0])**2 +
            (pos2[1] - pos1[1])**2 +
            (pos2[2] - pos1[2])**2
        )

    def _start_movement(self, wait: bool = False):
        self._stop_movement = True
        if self._movement_thread and self._movement_thread.is_alive():
            self._movement_thread.join()

        self._stop_movement = False
        self._state = self.STATE_MOVING

        self._movement_thread = threading.Thread(
            target=self._simulate_movement)
        self._movement_thread.daemon = True
        self._movement_thread.start()

        if wait:
            self._movement_thread.join()

    def _simulate_movement(self):
        start_pos = self._current_position.copy()
        target_pos = self._target_position.copy()

        distance = self._calculate_distance(start_pos, target_pos)
        if distance < 0.1:
            self._state = self.STATE_READY
            return

        move_time = distance / self._current_speed
        steps = int(move_time * 50)
        steps = max(steps, 10)

        delay = move_time / steps

        for step in range(steps + 1):
            if self._stop_movement:
                print("XArmAPI Emulator: movement stopped.")
                break

            t = step / steps
            with self._movement_lock:
                for i in range(6):
                    self._current_position[i] = start_pos[i] + \
                        (target_pos[i] - start_pos[i]) * t

            time.sleep(delay)

        with self._movement_lock:
            for i in range(3):
                self._current_position[i] += random.uniform(-0.5, 0.5)

        self._total_distance += distance
        self._total_movements += 1

        self._state = self.STATE_READY
        print(
            f"XArmAPI Emulator: position reached {self._format_position(self._current_position)}.")

    def set_servo_angle(self, servo_id: Optional[int] = None, angle: Optional[float] = None,
                        speed: Optional[float] = None, wait: bool = False, **kwargs) -> int:
        print(f"XArmAPI Emulator: set servo={servo_id}, angle={angle}°")
        if wait:
            time.sleep(0.5)
        return self.CODE_SUCCESS

    def set_gripper_position(self, pos: float, wait: bool = False, speed: Optional[float] = None,
                             **kwargs) -> int:
        print(f"XArmAPI Emulator: set gripper position {pos}")
        if wait:
            time.sleep(0.5)
        return self.CODE_SUCCESS

    def set_vacuum_gripper(self, on, wait=False, timeout=3, delay_sec=None, sync=True, hardware_version=1) -> int:
        print(f"XArmAPI Emulator: set vacuum gripper to {on}")
        if wait:
            time.sleep(0.5)
        return self.CODE_SUCCESS

    def get_vacuum_gripper(self, hardware_version=1) -> int:
        print(f"XArmAPI Emulator: get vacuum gripper")
        time.sleep(0.5)
        return 1

    def emergency_stop(self) -> int:
        self._stop_movement = True
        self._state = self.STATE_ERROR
        print("XArmAPI Emulator: emergency stop.")
        return self.CODE_SUCCESS

    def clean_error(self) -> int:
        self._error_code = 0
        self._state = self.STATE_READY
        print("XArmAPI Emulator: errors are cleaned.")
        return self.CODE_SUCCESS

    def get_err_warn_code(self) -> Tuple[int, List[int]]:
        return (self.CODE_SUCCESS, [self._error_code, 0])

    def get_version(self) -> Tuple[int, str]:
        return (self.CODE_SUCCESS, "v0.0.1-emulator")

    def get_statistics(self) -> dict:
        return {
            'total_distance': self._total_distance,
            'total_movements': self._total_movements,
            'current_position': self._current_position.copy(),
            'state': self._state,
            'motion_enabled': self._motion_enabled
        }

    def reset_statistics(self):
        self._total_distance = 0.0
        self._total_movements = 0
        print("XArmAPI Emulator: statistics reset.")

    def reset(self, wait=True):
        print("XArmAPI Emulator: arm reset.")

    def set_gripper_mode(self, _):
        print("XArmAPI Emulator: arm set_gripper_mode.")

    def set_gripper_enable(self, _):
        print("XArmAPI Emulator: arm set_gripper_enable.")

    def __del__(self):
        self.disconnect()


# Wrapper
class XArmAPI(XArmEmulator):
    pass
