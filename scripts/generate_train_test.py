import bpy
import os
import math
from mathutils import Vector

# ===============
# CONFIGURATION
# ===============
DATASET_DIR = "/home/ashant/Desktop/project/tmp/blender_data"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")

NUM_TRAIN_CAMERAS = 160  # 80% of total data
NUM_TEST_CAMERAS = 40   # 20% for testing
CAMERA_RADIUS = 10

# ===============
# UTILITY FUNCTIONS
# ===============

def reset_scene():
    """Deletes all objects in the scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def compute_centroid(vertices):
    """Returns the centroid of a list of 3 vertices."""
    return (vertices[0] + vertices[1] + vertices[2]) / 3

def create_scalene_triangle():
    """Creates a scalene triangle centered at the origin."""
    bpy.ops.object.select_all(action='DESELECT')

    verts = [
        Vector((-1.0, -0.5, 0.0)),   # Vertex A
        Vector((1.0, -0.2, 0.0)),    # Vertex B
        Vector((0.0, 1.5, 0.0))      # Vertex C
    ]
    vertices = [tuple(v) for v in verts]
    edges = []
    faces = [[0,1,2]]

    mesh = bpy.data.meshes.new(name="ScaleneTriangle")
    obj = bpy.data.objects.new(name="ScaleneTriangle", object_data=mesh)

    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    mesh.from_pydata(vertices, edges, faces)
    mesh.update()

    print("✅ Scalene triangle created.")
    return obj, verts

def create_camera(location, index, target):
    """Creates a camera pointing toward a given target."""
    cam_data = bpy.data.cameras.new(name=f"Camera_{index}")
    cam_obj = bpy.data.objects.new(name=f"Camera_{index}", object_data=cam_data)
    cam_obj.location = location
    bpy.context.collection.objects.link(cam_obj)

    direction = target - location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam_obj.rotation_euler = rot_quat.to_euler()

    return cam_obj

def place_cameras(n=8, radius=5, target=Vector((0, 0, 0))):
    """Places N cameras uniformly around the object on a circle."""
    cameras = []
    for i in range(n):
        angle = 2 * math.pi * i / n
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        z = 2  # Slight elevation

        location = Vector((x, y, z))
        cam = create_camera(location, i, target)
        cameras.append(cam)

    return cameras

def setup_render_settings(output_path):
    """Configures render settings and output path."""
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'CPU'  # Safe default
    bpy.context.scene.cycles.use_denoising = False
    bpy.context.scene.cycles.samples = 32
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.resolution_x = 256
    bpy.context.scene.render.resolution_y = 256
    bpy.context.scene.render.filepath = output_path

def render_image(filepath, camera):
    """Renders image from given camera and saves it."""
    bpy.context.scene.camera = camera
    bpy.context.scene.render.filepath = filepath
    bpy.ops.render.render(write_still=True)

def add_three_point_lighting():
    """Adds key, fill, and back lights."""
    def create_light(name, type, location, energy):
        light_data = bpy.data.lights.new(name=name, type=type)
        light_data.energy = energy
        light = bpy.data.objects.new(name=name, object_data=light_data)
        bpy.context.collection.objects.link(light)
        light.location = location
        return light

    create_light("Key_Light", 'POINT', (5, -5, 5), 1000)
    create_light("Fill_Light", 'POINT', (-4, -4, 2), 500)
    create_light("Back_Light", 'POINT', (0, 5, 5), 700)
    print("💡 Three-point lighting added.")

# ===============
# DATASET GENERATOR
# ===============

def generate_dataset(output_dir, num_cameras=20):
    print(f"🚀 Generating dataset in {output_dir}...")

    reset_scene()
    add_three_point_lighting()

    triangle_obj, vertices = create_scalene_triangle()
    centroid = compute_centroid(vertices)

    cameras = place_cameras(n=num_cameras, radius=CAMERA_RADIUS, target=centroid)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 Created directory: {output_dir}")

    metadata_path = os.path.join(output_dir, "metadata.csv")
    with open(metadata_path, 'w') as f:
        f.write("camera_index,image_path,distance\n")

        for i, cam in enumerate(cameras):
            img_path = os.path.join(output_dir, f"image_{i}.png")
            dist = (cam.location - centroid).length
            setup_render_settings(img_path)
            print(f"📸 Rendering image {i} | Distance: {dist:.2f}")
            render_image(img_path, cam)
            f.write(f"{i},{img_path},{dist:.4f}\n")

    print(f"✅ Dataset saved to {output_dir}")

# ===============
# MAIN FUNCTION
# ===============

if __name__ == "__main__":
    generate_dataset(TRAIN_DIR, NUM_TRAIN_CAMERAS)
    generate_dataset(TEST_DIR, NUM_TEST_CAMERAS)