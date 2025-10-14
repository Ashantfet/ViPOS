import bpy
import os
import math
import random
from mathutils import Vector

# ========== CONFIG ==========
SAVE_DIR = "/home/ashant/Desktop/project/tmp/blender_data_final_render"
os.makedirs(SAVE_DIR, exist_ok=True)

NUM_CAMERAS = 100  # Adjust as needed
TRIANGLE_RADIUS = 2.0 # This radius will now be less directly applied to vertex positions for a scalene triangle
SPHERE_RADIUS = 0.2
CAMERA_DISTANCE = 10.0

# ========== SETUP UTILS ==========
def clear_scene():
    """Clears all objects from the current Blender scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def create_triangle(radius):
    """
    Creates a scalene triangle mesh with a solidify modifier and a material.
    Ensures normals are correctly calculated for proper rendering.
    The 'radius' parameter is now less directly used for vertex positioning
    as the triangle is no longer equilateral.
    """
    # Define vertices for a scalene triangle
    # These coordinates are chosen to create unequal side lengths and angles.
    verts = [
        (0.0, radius, 0.0),            # Top vertex
        (-radius * 0.8, -radius * 0.5, 0.0), # Bottom-left vertex
        (radius * 1.2, -radius * 0.3, 0.0)   # Bottom-right vertex
    ]

    mesh = bpy.data.meshes.new("ScaleneTriangle") # Changed mesh name
    obj = bpy.data.objects.new("ScaleneTriangle", mesh) # Changed object name
    bpy.context.collection.objects.link(obj)

    # Define face
    faces = [(0, 1, 2)]
    mesh.from_pydata(verts, [], faces)
    
    # IMPORTANT: Update mesh data
    mesh.update()

    # Set the object as active and select it
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # Switch to Edit Mode, perform the normal consistency operation, then switch back to Object Mode
    bpy.ops.object.mode_set(mode='EDIT')
    # Removed 'keep_original=False' as it's no longer a valid argument in Blender 4.0.2
    bpy.ops.mesh.normals_make_consistent()
    bpy.ops.object.mode_set(mode='OBJECT')
    
    obj.select_set(False) # Deselect after operation

    # Add solidify modifier for thickness
    mod = obj.modifiers.new("Solidify", type='SOLIDIFY')
    mod.thickness = 0.02
    bpy.ops.object.modifier_apply(modifier="Solidify")

    # Add material
    mat = bpy.data.materials.new("Triangle_Material")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        # Made the base color slightly brighter for better visibility of edges
        bsdf.inputs["Base Color"].default_value = (0.6, 0.8, 0.8, 1) # Lighter cyan-ish
        bsdf.inputs["Roughness"].default_value = 0.5
        
        # --- NEW MATERIAL PROPERTIES FOR BETTER EDGE VISIBILITY ---
        bsdf.inputs["Metallic"].default_value = 0.1 # Add a subtle metallic sheen
        # Removed 'Specular' and 'Clearcoat' as they are no longer direct inputs in Blender 4.0.2
        # bsdf.inputs["Clearcoat"].default_value = 0.1 # Add a subtle clear coat
        # bsdf.inputs["Clearcoat Roughness"].default_value = 0.1 # Make clear coat slightly rough
        
        # Add a very small emission to ensure edges are never completely dark
        emission_node = mat.node_tree.nodes.new(type='ShaderNodeEmission')
        emission_node.inputs['Color'].default_value = (1, 1, 1, 1) # White emission
        emission_node.inputs['Strength'].default_value = 0.01 # Very subtle self-illumination
        
        # Connect emission to material output
        mat.node_tree.links.new(emission_node.outputs['Emission'], mat.node_tree.nodes['Material Output'].inputs['Surface'])
        # Connect Principled BSDF to emission, then emission to output (or mix shaders)
        # For simplicity, let's mix the Principled BSDF with the Emission shader
        mix_shader = mat.node_tree.nodes.new(type='ShaderNodeMixShader')
        mat.node_tree.links.new(bsdf.outputs['BSDF'], mix_shader.inputs[1])
        mat.node_tree.links.new(emission_node.outputs['Emission'], mix_shader.inputs[2])
        mat.node_tree.links.new(mix_shader.outputs['Shader'], mat.node_tree.nodes['Material Output'].inputs['Surface'])
        mix_shader.inputs['Fac'].default_value = 0.9 # Mostly Principled BSDF, a little emission
        
    obj.data.materials.append(mat)

    return obj, [Vector(v) for v in verts]


def add_colored_sphere(loc, color, name):
    """Adds a UV sphere with a specified color and name."""
    bpy.ops.mesh.primitive_uv_sphere_add(radius=SPHERE_RADIUS, location=loc)
    sphere = bpy.context.active_object
    sphere.name = name

    mat = bpy.data.materials.new(name + "_Mat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs.get("Base Color", None).default_value = color
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = 0.6
        if "Specular" in bsdf.inputs:
            bsdf.inputs["Specular"].default_value = 0.2

    sphere.data.materials.append(mat)


def add_light():
    """Adds sun light sources to the scene."""
    # First sun light (existing)
    bpy.ops.object.light_add(type='SUN', location=(5, 5, 5))
    light1 = bpy.context.active_object
    light1.data.energy = 10.0

    # --- NEW: Add a second sun light from a different angle for better fill ---
    bpy.ops.object.light_add(type='SUN', location=(-5, -5, 5)) # From opposite side
    light2 = bpy.context.active_object
    light2.data.energy = 5.0 # Slightly less intense fill light
    # Corrected: Set rotation on the object itself, not its data block
    light2.rotation_euler = (math.radians(30), math.radians(150), math.radians(0)) # Rotate to point towards origin


def add_camera_on_sphere(index, total, radius, target=Vector((0, 0, 0))):
    """
    Adds a camera on a sphere around the origin, looking at the origin.
    Cameras are distributed in layers based on index.
    """
    theta = 2 * math.pi * (index % (total // 4)) / (total // 4)
    phi = math.radians(30 + 60 * (index // (total // 4)) / 3.0)  # 3 layers of height

    x = radius * math.sin(phi) * math.cos(theta)
    y = radius * math.sin(phi) * math.sin(theta)
    z = radius * math.cos(phi)

    bpy.ops.object.camera_add(location=(x, y, z))
    cam = bpy.context.active_object
    cam.name = f"Camera_{index}"
    cam.data.lens = 18

    # Rotate to look at the origin
    direction = target - cam.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam.rotation_euler = rot_quat.to_euler()

    return cam, Vector((x, y, z))

def render_image(cam, index, cam_loc, metadata_file):
    """
    Renders an image from the specified camera and saves its metadata.
    """
    bpy.context.scene.camera = cam
    img_path = os.path.join(SAVE_DIR, f"image_{index:03d}.png")
    bpy.context.scene.render.filepath = img_path
    bpy.ops.render.render(write_still=True)

    # Calculate Azimuth & Elevation for metadata
    direction = Vector((0, 0, 0)) - cam_loc
    azimuth = math.degrees(math.atan2(direction.y, direction.x))
    hyp = math.sqrt(direction.x**2 + direction.y**2)
    elevation = math.degrees(math.atan2(direction.z, hyp))

    metadata_file.write(f"{index},{img_path},{cam_loc.x:.4f},{cam_loc.y:.4f},{cam_loc.z:.4f},{azimuth:.2f},{elevation:.2f}\n")

# ========== MAIN ==========
def main():
    """Main function to set up the scene, add objects, and render images."""
    clear_scene()
    triangle, verts = create_triangle(TRIANGLE_RADIUS)

    # Use accurate red, green, and blue spheres
    # The sphere positions are now based on the new scalene triangle vertices
    add_colored_sphere(verts[0], (0.3, 0.0, 0.0, 1), "Red")     # Accurate deep red
    add_colored_sphere(verts[1], (0.0, 0.3, 0.0, 0), "Green")
    add_colored_sphere(verts[2], (0.0, 0.0, 0.3, 0), "Blue")

    add_light()

    # Render settings
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 64
    scene.cycles.use_denoising = False
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.image_settings.file_format = 'PNG'

    # Metadata file
    metadata_path = os.path.join(SAVE_DIR, "metadata.csv")
    with open(metadata_path, "w") as f:
        f.write("index,image_path,x,y,z,azimuth,elevation\n")
        for i in range(NUM_CAMERAS):
            cam, cam_loc = add_camera_on_sphere(i, NUM_CAMERAS, CAMERA_DISTANCE)
            render_image(cam, i, cam_loc, f)
            bpy.data.objects.remove(cam, do_unlink=True) # Clean up camera after rendering

    print("✅ Dataset rendered to:", SAVE_DIR)

# ========== ENTRY ==========
if __name__ == "__main__":
    main()
