from controller import Supervisor
import random
import os

CAMERA_NAME = "camera"

IMAGES_COUNT = 12

NUM_OBJECTS = 30
PROTO_NAMES = ["Plate1_bin", "Plate2_bin", "Plate3_bin",
               "Plate4_bin", "Plate5_bin"]
X_RANGE = (-0.25, 0.25)
Y_RANGE = (-0.10, 0.10)
Z_RANGE = (1.50, 1.70)
SPAWN_INTERVAL = 150


class BinSpawner:
    def __init__(self, supervisor):
        self.supervisor = supervisor
        self.spawned = 0
        self.step_count = 0

    def random_pose(self):
        x = random.uniform(*X_RANGE)
        y = random.uniform(*Y_RANGE)
        z = random.uniform(*Z_RANGE)
        rot_axis = [random.random() for _ in range(3)]
        rot_angle = random.uniform(0, 3.14)
        return x, y, z, rot_axis + [rot_angle]

    def spawn_object(self):
        proto_name = random.choice(PROTO_NAMES)
        x, y, z, rotation = self.random_pose()

        proto_string = f'DEF SP_OBJ_{self.spawned} {proto_name} {{ translation {x} {y} {z} rotation {" ".join(map(str, rotation))} }}'
        root = self.supervisor.getRoot()
        children_field = root.getField("children")
        children_field.importMFNodeFromString(-1, proto_string)

        print(f"Spawned: {proto_name} at ({x:.2f}, {y:.2f}, {z:.2f})")
        self.spawned += 1

    def run(self, timestep):
        self.spawned = 0
        self.step_count = 0
        while self.supervisor.step(timestep) != -1:
            if self.spawned < NUM_OBJECTS and self.step_count % SPAWN_INTERVAL == 0:
                self.spawn_object()
            self.step_count += 1
            if self.spawned == NUM_OBJECTS and self.step_count % SPAWN_INTERVAL == 0:
                break

    def delete_spawned_objects(self):
        root_children = self.supervisor.getRoot().getField("children")
        i = 0
        while i < root_children.getCount():
            node = root_children.getMFNode(i)
            node_def = node.getDef()
            if node_def.startswith("SP_OBJ"):
                print(f"Removing {node_def}")
                root_children.removeMF(i)
            else:
                i += 1


if not os.path.exists("output"):
    os.makedirs("output")

supervisor = Supervisor()
timestep = int(supervisor.getBasicTimeStep())
camera = supervisor.getDevice(CAMERA_NAME)
camera.enable(100)
spawner = BinSpawner(supervisor)

iteration = 0

while iteration < IMAGES_COUNT:
    spawner.run(timestep)
    supervisor.getFromDef("SPOT_1").getField("on").setSFBool(False)
    supervisor.getFromDef("SPOT_2").getField("on").setSFBool(False)
    supervisor.step(300)
    camera.getImageArray()
    camera.saveImage(f"output/image_{iteration:03d}_00.jpg", 100)
    
    supervisor.getFromDef("SPOT_1").getField("on").setSFBool(True)
    supervisor.getFromDef("SPOT_2").getField("on").setSFBool(False)
    supervisor.step(300)
    camera.getImageArray()
    camera.saveImage(f"output/image_{iteration:03d}_10.jpg", 100)
    
    supervisor.getFromDef("SPOT_1").getField("on").setSFBool(False)
    supervisor.getFromDef("SPOT_2").getField("on").setSFBool(True)
    supervisor.step(300)
    camera.getImageArray()
    camera.saveImage(f"output/image_{iteration:03d}_01.jpg", 100)
    
    supervisor.getFromDef("SPOT_1").getField("on").setSFBool(True)
    supervisor.getFromDef("SPOT_2").getField("on").setSFBool(True)
    supervisor.step(300)
    camera.getImageArray()
    camera.saveImage(f"output/image_{iteration:03d}_11.jpg", 100)
    iteration += 1
    spawner.delete_spawned_objects()
