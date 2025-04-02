from controller import Supervisor

import random


supervisor = Supervisor()

timestep = int(supervisor.getBasicTimeStep())

receiver = supervisor.getDevice("receiver")
receiver.enable(50)


OBJECT_NAMES = ["Plate1", "Plate2", "Plate3", "Plate4", "Plate5"]
TABLE_BOUNDS_X = (0.1, 0.15)
TABLE_BOUNDS_Y = (0.1, 0.15)


object_nodes = list(map(lambda name: supervisor.getFromDef(name), OBJECT_NAMES))
object_translation_fields = list(map(
    lambda node: node.getField("translation"), object_nodes))
object_rotation_fields = list(map(
    lambda node: node.getField("rotation"), object_nodes))


def random_pose(delta):
    dx, dy = delta
    x = dx * random.uniform(*TABLE_BOUNDS_X)
    y = dy * random.uniform(*TABLE_BOUNDS_Y)
    z = 0.885
    yaw = random.uniform(0, 6.28)
    return (x, y, z, yaw)


objects_order = list(range(len(OBJECT_NAMES)))
deltas = [(-1, 1), (1, 1), (1, -1), (-1, -1)]

while supervisor.step(timestep) != -1:
    if receiver.getQueueLength() > 0:
        message = receiver.getString()
        receiver.nextPacket()

        if message.startswith("reposition"):
            random.shuffle(objects_order)

            for i in range(4):
                x, y, z, yaw = random_pose(deltas[i])
                object_translation_fields[objects_order[i]].setSFVec3f([
                                                                       x, y, z])
                object_rotation_fields[objects_order[i]
                                       ].setSFRotation([0, 0, 1, yaw])

            for i in range(4, len(objects_order)):
                object_translation_fields[objects_order[i]].setSFVec3f(
                    [-0.34, 0.8, 0.7])
                object_rotation_fields[objects_order[i]
                                       ].setSFRotation([0, 0, 1, 0])
