"""simple_pick_and_place controller."""


from controller import Robot
from spatialmath import SE3
import roboticstoolbox as rtb

robot = Robot()
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
timestep = int(robot.getBasicTimeStep())

robot_kinematics = rtb.models.UR10()
robot_kinematics = rtb.models.UR10()


def get_solution(x, y, z):
    Tep = SE3.Trans(x, y, z) * SE3.Eul([-90, -90, -180], unit='deg')
    i = 0
    while i < 100:
        sol = robot_kinematics.ikine_GN(Tep)
        if sol.q[1] > -1.57 and sol.q[1] < 0 and sol.q[2] > 0 and sol.q[3] < 0:
            return sol.q
        i = i + 1
    return None


def go_to_position(x, y, z, fast=False):
    joint_angles = get_solution(x, y, z)
    if not fast:
        joint_angles[1] = joint_angles[1] - 0.2
        for motor, angle in zip(motors, joint_angles):
            motor.setPosition(angle)
        robot.step(100 * timestep)
        joint_angles[1] = joint_angles[1] + 0.2

    for motor, angle in zip(motors, joint_angles):
        motor.setPosition(angle)
    robot.step(100 * timestep)


go_to_position(0.5, 0.5, 0.6, fast=True)
gripper.turnOff()
robot.step(10 * timestep)

go_to_position(-1, -0.15, 0.164)
gripper.turnOn()
robot.step(10 * timestep)
go_to_position(-1, -0.15, 0.3, fast=True)

go_to_position(0, 0.8, 0.03)
gripper.turnOff()
robot.step(10 * timestep)

go_to_position(-1, 0, 0.164)
gripper.turnOn()
robot.step(10 * timestep)
go_to_position(-1, 0, 0.3, fast=True)

go_to_position(0, 1, 0.03)
gripper.turnOff()
robot.step(10 * timestep)

go_to_position(0.5, 0.5, 0.6, fast=True)
