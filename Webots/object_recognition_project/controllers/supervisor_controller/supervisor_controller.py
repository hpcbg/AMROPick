from controller import Supervisor

import random


supervisor = Supervisor()

timestep = int(supervisor.getBasicTimeStep())

receiver = supervisor.getDevice("receiver")
receiver.enable(50)

plate_1_node = supervisor.getFromDef("Plate1")
translation_field_1 = plate_1_node.getField("translation")
rotation_field_1 = plate_1_node.getField("rotation")

plate_3_node = supervisor.getFromDef("Plate3")
translation_field_3 = plate_3_node.getField("translation")
rotation_field_3 = plate_3_node.getField("rotation")

TABLE_BOUNDS_X = (0.1, 0.15)
TABLE_BOUNDS_Y = (-0.15, 0.15)


def random_pose(dir=1):
    x = dir * random.uniform(*TABLE_BOUNDS_X)
    y = random.uniform(*TABLE_BOUNDS_Y)
    z = 0.885
    yaw = random.uniform(0, 6.28)
    return (x, y, z, yaw)


iteration = 0

while supervisor.step(timestep) != -1:
    if receiver.getQueueLength() > 0:
        message = receiver.getString()
        receiver.nextPacket()

        if message.startswith("reposition"):
            x, y, z, yaw = random_pose(1 if iteration % 2 == 0 else -1)
            translation_field_1.setSFVec3f([x, y, z])
            rotation_field_1.setSFRotation([0, 0, 1, yaw])

            x, y, z, yaw = random_pose(-1 if iteration % 2 == 0 else 1)
            translation_field_3.setSFVec3f([x, y, z])
            rotation_field_3.setSFRotation([0, 0, 1, yaw])

            iteration = iteration + 1
