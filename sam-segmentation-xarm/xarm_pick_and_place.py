import time
import json
import os

import utils.json_config

CONFIG = utils.json_config.load("config/xarm_pick_and_place.json")

if CONFIG['EMULATE_ROBOT']:
    from utils.xarm_emulator import XArmAPI
else:
    from xarm.wrapper import XArmAPI


ROBOT_IP = CONFIG['ROBOT_IP']
HOME_POS = CONFIG['HOME_POS']
GRIPPER_OPEN_POS = CONFIG['GRIPPER_OPEN_POS']
GRIPPER_CLOSE_POS = CONFIG['GRIPPER_CLOSE_POS']
APPROACH_HEIGHT = CONFIG['APPROACH_HEIGHT']
GRIPPER_TYPE = CONFIG['GRIPPER_TYPE']
PICK_FILE = 'pick_and_place.json'
STATUS_FILE = 'robot_status.json'
SLEEP_TIME = 1


arm = XArmAPI(ROBOT_IP)
arm.connect()
arm.motion_enable(True)
arm.set_mode(0)
arm.set_state(0)
arm.clean_error()
# arm.set_gripper_mode(0)
if GRIPPER_TYPE == 'vacuum':
    arm.set_vacuum_gripper(on=False, wait=True, timeout=3, hardware_version=2)
else:
    arm.set_gripper_enable(True)


def write_status(status, pos=None):
    """Write the status to JSON"""
    data = {"status": status, "pos": pos if pos else None}
    with open(STATUS_FILE, 'w') as f:
        json.dump(data, f, indent=2)


def move_to_xyzrpy(x, y, z, roll=180, pitch=0, yaw=0, wait=True, speed=100, mvacc=100, status='OK'):
    """Move to XYZ RPY"""
    arm.set_position(x, y, z, roll, pitch, yaw, wait=wait,
                     speed=9*speed, mvacc=5*mvacc)
    write_status(status, {"x": x, "y": y, "z": z})


def pick_and_place(pick_and_place):
    try:
        pick_pose = pick_and_place['pick_pose']
        place_pose = pick_and_place_data['place_pose']

        x, y, z, roll, pitch, yaw = [204.6, 0, 346.1, 179.9, 0, 0]

        print("Open gripper")
        if GRIPPER_TYPE == 'vacuum':
            arm.set_vacuum_gripper(on=False, wait=True,
                                   timeout=3, hardware_version=2)
        else:
            arm.set_gripper_position(GRIPPER_OPEN_POS, wait=True)

        print("Move to home:", x, y, z, roll, pitch, yaw)
        move_to_xyzrpy(x, y, z, roll, pitch, yaw, status='BUSY')

        z = APPROACH_HEIGHT
        print("Move to approach height:", x, y, z, roll, pitch, yaw)
        move_to_xyzrpy(x, y, z, roll, pitch, yaw, status='BUSY')

        roll = pick_pose['roll_degrees']
        pitch = pick_pose['pitch_degrees']
        yaw = pick_pose['yaw_degrees']
        print("Move to pick orientation:", x, y, z, roll, pitch, yaw)
        move_to_xyzrpy(x, y, z, roll, pitch, yaw, status='BUSY')

        x = pick_pose['x']
        y = pick_pose['y']
        print("Move to pick XY position:", x, y, z, roll, pitch, yaw)
        move_to_xyzrpy(x, y, z, roll, pitch, yaw, status='BUSY')

        z = pick_pose['z']
        print("Move to pick Z position:", x, y, z, roll, pitch, yaw)
        move_to_xyzrpy(x, y, z, roll, pitch, yaw, status='BUSY')

        print("Close gripper")
        if GRIPPER_TYPE == 'vacuum':
            arm.set_vacuum_gripper(
                on=True, wait=True, timeout=3, hardware_version=2)
            _, grasped = arm.get_vacuum_gripper(hardware_version=2)
            retries = 3
            while grasped != 1 and retries > 0:
                retries = retries - 1
                arm.set_vacuum_gripper(
                    on=False, wait=True, timeout=5, hardware_version=2)
                z = z - 5
                print("Move to pick Z position:", x, y, z, roll, pitch, yaw)
                move_to_xyzrpy(x, y, z, roll, pitch, yaw, status='BUSY')
                arm.set_vacuum_gripper(
                    on=True, wait=True, timeout=5, hardware_version=2)
                _, grasped = arm.get_vacuum_gripper(hardware_version=2)
                print("grasped", grasped)
        else:
            arm.set_gripper_position(GRIPPER_OPEN_POS, wait=True)

        z = APPROACH_HEIGHT
        print("Move to approach height:", x, y, z, roll, pitch, yaw)
        move_to_xyzrpy(x, y, z, roll, pitch, yaw, status='BUSY')

        roll = place_pose['roll_degrees']
        pitch = place_pose['pitch_degrees']
        yaw = place_pose['yaw_degrees']
        print("Move to place orientation:", x, y, z, roll, pitch, yaw)
        move_to_xyzrpy(x, y, z, roll, pitch, yaw, status='BUSY')

        x = place_pose['x']
        y = place_pose['y']
        print("Move to place XY position:", x, y, z, roll, pitch, yaw)
        move_to_xyzrpy(x, y, z, roll, pitch, yaw, status='BUSY')

        z = place_pose['z']
        print("Move to place Z position:", x, y, z, roll, pitch, yaw)
        move_to_xyzrpy(x, y, z, roll, pitch, yaw, status='BUSY')

        print("Open gripper")
        if GRIPPER_TYPE == 'vacuum':
            arm.set_vacuum_gripper(on=False, wait=True,
                                   timeout=3, hardware_version=2)
        else:
            arm.set_gripper_position(GRIPPER_OPEN_POS, wait=True)

        z = APPROACH_HEIGHT
        print("Move to approach height:", x, y, z, roll, pitch, yaw)
        move_to_xyzrpy(x, y, z, roll, pitch, yaw, status='BUSY')

    except Exception as e:
        print("ERROR:", e)
    finally:
        # Return to home position
        print("Move to home:", HOME_POS)
        arm.set_servo_angle(angle=HOME_POS, wait=True)
        if GRIPPER_TYPE == 'vacuum':
            arm.set_vacuum_gripper(on=False, wait=True,
                                   timeout=3, hardware_version=2)
        else:
            arm.set_gripper_position(GRIPPER_OPEN_POS, wait=True)
        write_status(
            "OK", {"x": HOME_POS[0], "y": HOME_POS[1], "z": HOME_POS[2]})


write_status("BUSY", {"x": HOME_POS[0], "y": HOME_POS[1], "z": HOME_POS[2]})
if GRIPPER_TYPE == 'vacuum':
    arm.set_vacuum_gripper(on=False, wait=True, timeout=3, hardware_version=2)
else:
    arm.set_gripper_position(GRIPPER_OPEN_POS, wait=True)
arm.set_servo_angle(angle=HOME_POS, wait=True)
write_status("OK", {"x": HOME_POS[0], "y": HOME_POS[1], "z": HOME_POS[2]})
while True:
    if os.path.exists(PICK_FILE):
        try:
            with open(PICK_FILE, 'r') as f:
                pick_and_place_data = json.load(f)
            print("Pick and place requested:", pick_and_place_data)
            # Remove the file
            os.remove(PICK_FILE)
            # Execute pick and place
            pick_and_place(pick_and_place_data)

        except Exception as e:
            print("Error reading pick_and_place.json:", e)
    time.sleep(SLEEP_TIME)
