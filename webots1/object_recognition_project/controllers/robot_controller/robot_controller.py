from controller import Robot

from spatialmath import SE3
import roboticstoolbox as rtb
import math

robot = Robot()

timestep = int(robot.getBasicTimeStep())

motors = [
    robot.getDevice('shoulder_pan_joint'),
    robot.getDevice('shoulder_lift_joint'),
    robot.getDevice('elbow_joint'),
    robot.getDevice('wrist_1_joint'),
    robot.getDevice('wrist_2_joint'),
    robot.getDevice('wrist_3_joint')
]

gripper = robot.getDevice('gripper')
gripper.enablePresence(10)

receiver = robot.getDevice("receiver")
receiver.enable(50)
emitter = robot.getDevice("emitter")

robot_kinematics = rtb.models.UR10()

home = (0.5, 0.5, 0.6)
place = (0, 0.6, 0.03)
place_i = 0
place_step = 0.15


def get_solution(x, y, z, theta=0):
    Tep = SE3.Trans(x, y, z) * \
        SE3.Eul([-90 + math.degrees(theta), -90, -180], unit='deg')
    i = 0
    while i < 100:
        sol = robot_kinematics.ikine_GN(Tep)
        if sol.q[1] > -1.57 and sol.q[1] < 0 and sol.q[2] > 0 and sol.q[3] < 0:
            return sol.q
        i = i + 1
    return None


def go_to_position(x, y, z, theta=0, fast=False):
    joint_angles = get_solution(x, y, z, theta)
    if not fast:
        joint_angles[1] = joint_angles[1] - 0.2
        for motor, angle in zip(motors, joint_angles):
            motor.setPosition(angle)
        robot.step(50 * timestep)
        joint_angles[1] = joint_angles[1] + 0.2

    for motor, angle in zip(motors, joint_angles):
        motor.setPosition(angle)
    robot.step(50 * timestep)


while robot.step(timestep) != -1:
    if receiver.getQueueLength() > 0:
        message = receiver.getString()
        receiver.nextPacket()
        if message.startswith("go"):
            x, y, z = tuple(map(float, message.split("_")[1:]))
            go_to_position(x, y, z, fast=True)
            robot.step(10 * timestep)
        elif message.startswith("home"):
            x, y, z = home
            go_to_position(x, y, z, fast=True)
            robot.step(10 * timestep)
        elif message.startswith("reposition"):
            place_i = 0
        elif message.startswith("pick"):
            cx, cy, cz, ctheta = tuple(map(float, message.split("_")[1:]))
            x = -1.0 - cy
            y = cx
            z = 0.16 + cz
            theta = -ctheta

            gripper.turnOff()
            robot.step(10 * timestep)

            go_to_position(x, y, z)
            gripper.turnOn()
            robot.step(10 * timestep)
            go_to_position(x, y, z + 0.1, fast=True)

            x, y, z = place
            y = y + place_i * place_step
            place_i = place_i + 1
            go_to_position(x, y, z, theta)
            gripper.turnOff()
            robot.step(10 * timestep)

            x, y, z = home
            go_to_position(x, y, z, fast=True)
            robot.step(10 * timestep)
