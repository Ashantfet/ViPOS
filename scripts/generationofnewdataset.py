import bpy
import os
import math
from mathutils import Vector
import sys

# default mode
APRILTAG_MODE = "scale"

# check if passed via command-line
for arg in sys.argv:
    if arg.startswith("--mode="):
        APRILTAG_MODE = arg.split("=")[1]

# ===== CONFIG =====
SAVE_DIR = "/scratch/kalidas_1/project/tmp/blender_data_modes"
os.makedirs(os.path.join(SAVE_DIR, "images"), exist_ok=True)

NUM_CAMERAS = 30   # test small, then 100000
TRIANGLE_SCALE_FACTOR = 10.0
SPHERE_RADIUS = 1.0
CAMERA_DISTANCE = 100.0
APRILTAG_PATH = "/scratch/kalidas_1/project/tags/apriltag_36h11_00001.png"

# ===== CLEANUP =====
def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    for block in (bpy.data.meshes, bpy.data.cameras, bpy.data.lights, bpy.data.materials, bpy.data.images):
        for b in block:
            try:
                block.remove(b, do_unlink=True)
            except:
                pass

# ===== TRIANGLE WITH APRILTAG (SCALE MODE) =====
def create_triangle_with_scaled_apriltag(scale_factor, texture_path):
    base_verts = [(0.0, 4.0, 0.0), (0.0, 0.0, 0.0), (3.0, 0.0, 0.0)]
    scaled = [(x*scale_factor, y*scale_factor, z*scale_factor) for x,y,z in base_verts]
    cx = sum(v[0] for v in scaled)/3.0
    cy = sum(v[1] for v in scaled)/3.0
    cz = sum(v[2] for v in scaled)/3.0
    centered = [(x-cx, y-cy, z-cz) for x,y,z in scaled]

    mesh = bpy.data.meshes.new("ScaledAprilTagTriangle")
    mesh.from_pydata(centered, [], [(0,1,2)])
    mesh.update()
    obj = bpy.data.objects.new("ScaledAprilTagTriangle", mesh)
    bpy.context.collection.objects.link(obj)

    # Add UVs that map entire AprilTag square into triangle bounds
    uv_layer = mesh.uv_layers.new(name="UVMap")
    uv_coords = {
        0: (0.5, 1.0),  # top vertex maps to middle-top of tag
        1: (0.0, 0.0),  # left vertex -> bottom-left
        2: (1.0, 0.0),  # right vertex -> bottom-right
    }
    for poly in mesh.polygons:
        for loop_index in poly.loop_indices:
            vi = mesh.loops[loop_index].vertex_index
            uv_layer.data[loop_index].uv = uv_coords[vi]

    # Material with AprilTag
    mat = bpy.data.materials.new("AprilTag_Scaled")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    tex_node = mat.node_tree.nodes.new("ShaderNodeTexImage")
    tex_node.image = bpy.data.images.load(texture_path)
    mat.node_tree.links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])
    obj.data.materials.append(mat)

    return obj, [Vector(v) for v in centered]

# ===== APRILTAG PLANE + MASK (Mask Mode) =====
def create_apriltag_plane_with_mask(size, texture_path):
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0,0,0))
    plane = bpy.context.active_object
    plane.name = "AprilTagPlane"

    mat = bpy.data.materials.new("AprilTag_Masked")
    mat.use_nodes = True
    mat.blend_method = 'CLIP'
    mat.shadow_method = 'CLIP'
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for n in nodes:
        nodes.remove(n)

    tex_node = nodes.new("ShaderNodeTexImage")
    tex_node.image = bpy.data.images.load(texture_path)
    tex_node.interpolation = 'Closest'

    transp_node = nodes.new("ShaderNodeBsdfTransparent")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    mix_node = nodes.new("ShaderNodeMixShader")
    output = nodes.new("ShaderNodeOutputMaterial")

    bsdf.inputs["Base Color"].default_value = (1,1,1,1)
    links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(tex_node.outputs["Alpha"], mix_node.inputs["Fac"])
    links.new(transp_node.outputs["BSDF"], mix_node.inputs[1])
    links.new(bsdf.outputs["BSDF"], mix_node.inputs[2])
    links.new(mix_node.outputs["Shader"], output.inputs["Surface"])

    plane.data.materials.append(mat)
    return plane

def create_triangle_mask_with_edges(scale_factor):
    base_verts = [(0.0, 4.0, 0.0), (0.0, 0.0, 0.0), (3.0, 0.0, 0.0)]
    scaled = [(x*scale_factor, y*scale_factor, z*scale_factor) for x,y,z in base_verts]
    cx = sum(v[0] for v in scaled)/3.0
    cy = sum(v[1] for v in scaled)/3.0
    cz = sum(v[2] for v in scaled)/3.0
    centered = [(x-cx, y-cy, z-cz) for x,y,z in scaled]

    # Invisible mask triangle (for clipping only)
    mesh = bpy.data.meshes.new("MaskTriangle")
    mesh.from_pydata(centered, [], [(0,1,2)])
    mesh.update()
    mask_obj = bpy.data.objects.new("MaskTriangle", mesh)
    bpy.context.collection.objects.link(mask_obj)
    mask_obj.hide_render = True

    # Overlay wireframe triangle for visibility
    wire_obj = mask_obj.copy()
    wire_obj.data = mask_obj.data.copy()
    bpy.context.collection.objects.link(wire_obj)

    mod = wire_obj.modifiers.new("Wireframe", type='WIREFRAME')
    mod.thickness = 0.05 * scale_factor

    # Wireframe material
    mat = bpy.data.materials.new("TriangleEdges")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    bsdf = nodes["Principled BSDF"]

    # Set base color to white
    bsdf.inputs["Base Color"].default_value = (1, 1, 1, 1)

    # Add an emission node
    emission = nodes.new(type="ShaderNodeEmission")
    emission.inputs["Color"].default_value = (1, 1, 1, 1)
    emission.inputs["Strength"].default_value = 3.0

    # Mix BSDF and Emission
    mix_shader = nodes.new(type="ShaderNodeAddShader")
    links.new(bsdf.outputs["BSDF"], mix_shader.inputs[0])
    links.new(emission.outputs["Emission"], mix_shader.inputs[1])

    # Connect to material output
    output = nodes["Material Output"]
    links.new(mix_shader.outputs["Shader"], output.inputs["Surface"])


    wire_obj.data.materials.append(mat)

    return mask_obj, [Vector(v) for v in centered]

# ===== SPHERES =====
def add_colored_sphere(loc, color, name):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=SPHERE_RADIUS, location=loc)
    sphere = bpy.context.active_object
    mat = bpy.data.materials.new(name+"_Mat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*color,1)
    sphere.data.materials.append(mat)

# ===== LIGHTS =====
def add_lights():
    bpy.ops.object.light_add(type='SUN', location=(5,5,5))
    bpy.context.active_object.data.energy = 8
    bpy.ops.object.light_add(type='SUN', location=(-5,-5,5))
    bpy.context.active_object.data.energy = 5

# ===== CAMERA =====
def add_camera_on_sphere(i, total, radius, target=Vector((0,0,0))):
    phi = math.acos(1 - 2*(i+0.5)/total)
    theta = math.pi * (1 + 5**0.5) * (i+0.5)
    x = radius*math.sin(phi)*math.cos(theta)
    y = radius*math.sin(phi)*math.sin(theta)
    z = radius*math.cos(phi)
    bpy.ops.object.camera_add(location=(x,y,z))
    cam = bpy.context.active_object
    direction = (target - cam.location).normalized()
    rot_quat = direction.to_track_quat('-Z','Y')
    cam.rotation_mode = 'XYZ'
    cam.rotation_euler = rot_quat.to_euler()
    return cam, Vector((x,y,z))

# ===== RENDER =====
def render_image(cam, i, cam_loc, f):
    scene = bpy.context.scene
    scene.camera = cam
    path = os.path.join(SAVE_DIR, "images", f"image_{i:05d}.png")
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    tx,ty,tz = cam_loc
    roll, pitch, yaw = [math.degrees(a) for a in cam.rotation_euler]
    f.write(f"{i},{path},{tx:.4f},{ty:.4f},{tz:.4f},{roll:.2f},{pitch:.2f},{yaw:.2f}\n")

# ===== MAIN =====
def main():
    clear_scene()

    if APRILTAG_MODE == "scale":
        tri, verts = create_triangle_with_scaled_apriltag(TRIANGLE_SCALE_FACTOR, APRILTAG_PATH)
    else:
        create_apriltag_plane_with_mask(size=TRIANGLE_SCALE_FACTOR*5, texture_path=APRILTAG_PATH)
        tri, verts = create_triangle_mask_with_edges(TRIANGLE_SCALE_FACTOR)

    add_colored_sphere(verts[0], (1,0,0), "Red")
    add_colored_sphere(verts[1], (0,1,0), "Green")
    add_colored_sphere(verts[2], (0,0,1), "Blue")
    add_lights()

    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 16
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.image_settings.file_format = 'PNG'
    scene.render.film_transparent = True

    meta_path = os.path.join(SAVE_DIR, "metadata.csv")
    with open(meta_path,"w") as f:
        f.write("index,path,tx,ty,tz,roll,pitch,yaw\n")
        for i in range(NUM_CAMERAS):
            if i % 10 == 0: print(f"Rendering {i}/{NUM_CAMERAS}")
            cam, loc = add_camera_on_sphere(i, NUM_CAMERAS, CAMERA_DISTANCE)
            render_image(cam,i,loc,f)
            bpy.data.objects.remove(cam, do_unlink=True)

    print("✅ Done! Mode:", APRILTAG_MODE, "Images in:", os.path.join(SAVE_DIR,"images"))

if __name__ == "__main__":
    main()
