import bpy
import os
import math
import random
from mathutils import Vector

# ========== CONFIG ==========
BASE_DIR = "/home/ashant/Desktop/project/tmp/blender_data_final_render_split"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

NUM_CAMERAS = 100
TRAIN_SPLIT = 80  # First 80 for training
TRIANGLE_RADIUS = 2.0
SPHERE_RADIUS = 0.2
CAMERA_DISTANCE = 10.0

# ========== SETUP UTILS ==========
def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def create_triangle(radius):
    verts = [
        (0.0, radius, 0.0),
        (-radius * 0.8, -radius * 0.5, 0.0),
        (radius * 1.2, -radius * 0.3, 0.0)
    ]

    mesh = bpy.data.meshes.new("ScaleneTriangle")
    obj = bpy.data.objects.new("ScaleneTriangle", mesh)
    bpy.context.collection.objects.link(obj)

    faces = [(0, 1, 2)]
    mesh.from_pydata(verts, [], faces)
    mesh.update()

    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.normals_make_consistent()
    bpy.ops.object.mode_set(mode='OBJECT')
    obj.select_set(False)

    mod = obj.modifiers.new("Solidify", type='SOLIDIFY')
    mod.thickness = 0.02
    bpy.ops.object.modifier_apply(modifier="Solidify")

    mat = bpy.data.materials.new("Triangle_Material")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.6, 0.8, 0.8, 1)
        bsdf.inputs["Roughness"].default_value = 0.5
        bsdf.inputs["Metallic"].default_value = 0.1

        emission_node = mat.node_tree.nodes.new(type='ShaderNodeEmission')
        emission_node.inputs['Color'].default_value = (1, 1, 1, 1)
        emission_node.inputs['Strength'].default_value = 0.01

        mix_shader = mat.node_tree.nodes.new(type='ShaderNodeMixShader')
        mat.node_tree.links.new(bsdf.outputs['BSDF'], mix_shader.inputs[1])
        mat.node_tree.links.new(emission_node.outputs['Emission'], mix_shader.inputs[2])
        mat.node_tree.links.new(mix_shader.outputs['Shader'], mat.node_tree.nodes['Material Output'].inputs['Surface'])
        mix_shader.inputs['Fac'].default_value = 0.9

    obj.data.materials.append(mat)
    return obj, [Vector(v) for v in verts]

def add_colored_sphere(loc, color, name):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=SPHERE_RADIUS, location=loc)
    sphere = bpy.context.active_object
    sphere.name = name

    mat = bpy.data.materials.new(name + "_Mat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs.get("Base Color", None).default_value = color
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = 0.6
        if "Specular" in bsdf.inputs:
            bsdf.inputs["Specular"].default_value = 0.2
    sphere.data.materials.append(mat)

def add_light():
    bpy.ops.object.light_add(type='SUN', location=(5, 5, 5))
    light1 = bpy.context.active_object
    light1.data.energy = 10.0

    bpy.ops.object.light_add(type='SUN', location=(-5, -5, 5))
    light2 = bpy.context.active_object
    light2.data.energy = 5.0
    light2.rotation_euler = (math.radians(30), math.radians(150), math.radians(0))

def add_camera_on_sphere(index, total, radius, target=Vector((0, 0, 0))):
    theta = 2 * math.pi * (index % (total // 4)) / (total // 4)
    phi = math.radians(30 + 60 * (index // (total // 4)) / 3.0)

    x = radius * math.sin(phi) * math.cos(theta)
    y = radius * math.sin(phi) * math.sin(theta)
    z = radius * math.cos(phi)

    bpy.ops.object.camera_add(location=(x, y, z))
    cam = bpy.context.active_object
    cam.name = f"Camera_{index}"
    cam.data.lens = 18

    direction = target - cam.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam.rotation_euler = rot_quat.to_euler()

    return cam, Vector((x, y, z))

def render_image(cam, index, cam_loc, metadata_file, save_dir):
    bpy.context.scene.camera = cam
    img_path = os.path.join(save_dir, f"image_{index:03d}.png")
    bpy.context.scene.render.filepath = img_path
    bpy.ops.render.render(write_still=True)

    direction = Vector((0, 0, 0)) - cam_loc
    azimuth = math.degrees(math.atan2(direction.y, direction.x))
    hyp = math.sqrt(direction.x**2 + direction.y**2)
    elevation = math.degrees(math.atan2(direction.z, hyp))

    metadata_file.write(f"{index},{img_path},{cam_loc.x:.4f},{cam_loc.y:.4f},{cam_loc.z:.4f},{azimuth:.2f},{elevation:.2f}\n")

# ========== MAIN ==========
def main():
    clear_scene()
    triangle, verts = create_triangle(TRIANGLE_RADIUS)

    add_colored_sphere(verts[0], (0.3, 0.0, 0.0, 0), "Red")
    add_colored_sphere(verts[1], (0.0, 0.3, 0.0, 0), "Green")
    add_colored_sphere(verts[2], (0.0, 0.0, 0.3, 0), "Blue")

    add_light()

    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 64
    scene.cycles.use_denoising = False
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.image_settings.file_format = 'PNG'

    train_meta = open(os.path.join(TRAIN_DIR, "metadata.csv"), "w")
    test_meta = open(os.path.join(TEST_DIR, "metadata.csv"), "w")

    train_meta.write("index,image_path,x,y,z,azimuth,elevation\n")
    test_meta.write("index,image_path,x,y,z,azimuth,elevation\n")

    for i in range(NUM_CAMERAS):
        cam, cam_loc = add_camera_on_sphere(i, NUM_CAMERAS, CAMERA_DISTANCE)
        if i < TRAIN_SPLIT:
            render_image(cam, i, cam_loc, train_meta, TRAIN_DIR)
        else:
            render_image(cam, i, cam_loc, test_meta, TEST_DIR)
        bpy.data.objects.remove(cam, do_unlink=True)

    train_meta.close()
    test_meta.close()

    print("✅ Dataset rendered to:", BASE_DIR)

# ========== ENTRY ==========
if __name__ == "__main__":
    main()
