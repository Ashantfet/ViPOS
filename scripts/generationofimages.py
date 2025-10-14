import bpy
import os
import math
import random
from mathutils import Vector

# ========== CONFIGURATION ==========
# Directory to save rendered images and metadata
SAVE_DIR = "/scratch/kalidas_1/project/tmp/blender_data_final_render"
os.makedirs(SAVE_DIR, exist_ok=True)

# Number of camera views (images) to render
NUM_CAMERAS = 100000  # Set to 100,000 as requested

# Scale factor for the 3-4-5 right-angle triangle
TRIANGLE_SCALE_FACTOR = 10.0 # Scales the base (3,4,5) triangle, e.g., 10 -> (30,40,50) sides

# Radius of the small spheres at triangle vertices
SPHERE_RADIUS = 1.0 # Increased for better visibility with larger scene scale

# Distance of cameras from the origin (radius of the camera sphere)
CAMERA_DISTANCE = 100.0 # Set to 100 as requested

# ========== SCENE SETUP UTILITIES ==========
def clear_scene():
    """Clears all objects from the current Blender scene."""
    # Select all objects
    bpy.ops.object.select_all(action='SELECT')
    # Delete selected objects
    bpy.ops.object.delete()

def create_triangle(scale_factor):
    """
    Creates a right-angle scalene triangle mesh with (3,4,5) side lengths,
    scaled by 'scale_factor'. Adds a solidify modifier and a material.
    Ensures normals are correctly calculated for proper rendering.
    """
    # Define vertices for a base 3-4-5 right triangle.
    # The right angle is at (0,0,0).
    # Side lengths: 3 (along X), 4 (along Y), 5 (hypotenuse between (0,4,0) and (3,0,0))
    base_verts = [
        (0.0, 4.0, 0.0),            # Vertex A (y-axis)
        (0.0, 0.0, 0.0),            # Vertex B (origin - right angle)
        (3.0, 0.0, 0.0)             # Vertex C (x-axis)
    ]

    # Apply the scaling factor to all vertices
    scaled_verts = [(v[0] * scale_factor, v[1] * scale_factor, v[2] * scale_factor) for v in base_verts]

    # Create a new mesh and object
    mesh = bpy.data.meshes.new("RightAngleScaleneTriangle")
    obj = bpy.data.objects.new("RightAngleScaleneTriangle", mesh)
    bpy.context.collection.objects.link(obj)

    # Define the face (triangle)
    faces = [(0, 1, 2)]
    # Populate the mesh with vertices and faces
    mesh.from_pydata(scaled_verts, [], faces)
    
    # IMPORTANT: Update mesh data to apply changes
    mesh.update()

    # Set the object as active and select it for modifier operations
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # Switch to Edit Mode, ensure normals are consistent, then switch back to Object Mode
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.normals_make_consistent() # Ensures all face normals point outwards
    bpy.ops.object.mode_set(mode='OBJECT')
    
    obj.select_set(False) # Deselect the object after operations

    # Add Solidify modifier to give the triangle thickness
    mod = obj.modifiers.new("Solidify", type='SOLIDIFY')
    mod.thickness = 0.02 * scale_factor # Scale thickness proportionally
    bpy.ops.object.modifier_apply(modifier="Solidify")

    # Add a new material to the triangle object
    mat = bpy.data.materials.new("Triangle_Material")
    mat.use_nodes = True # Enable node-based material editing
    bsdf = mat.node_tree.nodes.get("Principled BSDF") # Get the default Principled BSDF shader node

    if bsdf:
        # Set base color to a light cyan-ish for good visibility
        bsdf.inputs["Base Color"].default_value = (0.6, 0.8, 0.8, 1)
        bsdf.inputs["Roughness"].default_value = 0.5
        bsdf.inputs["Metallic"].default_value = 0.1 # Subtle metallic sheen

        # Add a subtle emission to ensure edges are visible even in challenging lighting
        emission_node = mat.node_tree.nodes.new(type='ShaderNodeEmission')
        emission_node.inputs['Color'].default_value = (1, 1, 1, 1) # White emission
        emission_node.inputs['Strength'].default_value = 0.01 # Very subtle self-illumination
        
        # Mix the Principled BSDF and Emission shaders
        mix_shader = mat.node_tree.nodes.new(type='ShaderNodeMixShader')
        mat.node_tree.links.new(bsdf.outputs['BSDF'], mix_shader.inputs[1])
        mat.node_tree.links.new(emission_node.outputs['Emission'], mix_shader.inputs[2])
        
        # Connect the mixed shader output to the Material Output's Surface input
        mat.node_tree.links.new(mix_shader.outputs['Shader'], mat.node_tree.nodes['Material Output'].inputs['Surface'])
        mix_shader.inputs['Fac'].default_value = 0.9 # Mostly Principled BSDF, a small amount of emission
        
    obj.data.materials.append(mat) # Assign the created material to the object's mesh data

    # Return the created object and its scaled vertex positions as mathutils.Vector objects
    return obj, [Vector(v) for v in scaled_verts]


def add_colored_sphere(loc, color, name):
    """Adds a UV sphere with a specified color and name at a given location."""
    bpy.ops.mesh.primitive_uv_sphere_add(radius=SPHERE_RADIUS, location=loc)
    sphere = bpy.context.active_object
    sphere.name = name

    mat = bpy.data.materials.new(name + "_Mat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    if bsdf:
        # Set the base color using the provided RGB values (with alpha=1 for opaque)
        bsdf.inputs.get("Base Color", None).default_value = color + (1,) # Ensure alpha is 1
        bsdf.inputs["Roughness"].default_value = 0.6
        # The 'Specular' input was removed or changed in recent Blender versions.
        # Removing this line to prevent KeyError. Specular behavior is now often
        # controlled by 'IOR' (Index of Refraction) or 'Specular Tint'.
        # bsdf.inputs["Specular"].default_value = 0.2

    sphere.data.materials.append(mat)


def add_light():
    """Adds sun light sources to the scene for better illumination."""
    # Main sun light
    bpy.ops.object.light_add(type='SUN', location=(5, 5, 5))
    light1 = bpy.context.active_object
    light1.data.energy = 10.0 # Stronger light

    # Secondary sun light from a different angle for fill lighting
    bpy.ops.object.light_add(type='SUN', location=(-5, -5, 5))
    light2 = bpy.context.active_object
    light2.data.energy = 5.0 # Less intense fill light
    # Rotate the second light to point towards the origin
    light2.rotation_euler = (math.radians(30), math.radians(150), math.radians(0))


def add_camera_on_sphere(index, total, radius, target=Vector((0, 0, 0))):
    """
    Adds a camera on a sphere around the origin using a Golden Spiral distribution.
    This distribution provides a more even spread of cameras over the sphere's surface.
    The camera is rotated to look at the target point (origin).
    """
    # Golden Spiral distribution for even camera placement
    # Phi (elevation angle) ranges from 0 to pi
    phi = math.acos(1 - 2 * index / total)
    # Theta (azimuth angle) uses the golden angle to create a spiral
    theta = math.pi * (1 + 5**0.5) * index

    # Convert spherical coordinates to Cartesian coordinates
    x = radius * math.sin(phi) * math.cos(theta)
    y = radius * math.sin(phi) * math.sin(theta)
    z = radius * math.cos(phi)

    # Add the camera object to the scene
    bpy.ops.object.camera_add(location=(x, y, z))
    cam = bpy.context.active_object
    cam.name = f"Camera_{index:05d}" # Format with leading zeros for sorting
    cam.data.lens = 18 # Set camera focal length

    # Rotate the camera to look at the target (origin)
    direction = target - cam.location
    # Create a quaternion that tracks the -Z axis (camera's forward) towards the direction
    # and keeps the Y axis (camera's up) aligned with global Y.
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam.rotation_euler = rot_quat.to_euler() # Convert quaternion to Euler angles for assignment

    return cam, Vector((x, y, z))

def render_image(cam, index, cam_loc, metadata_file, camera_distance_max):
    """
    Renders an image from the specified camera and saves its metadata.
    Metadata includes normalized camera coordinates (x,y,z) and angles (azimuth, elevation).
    """
    bpy.context.scene.camera = cam # Set the current camera for rendering
    img_path = os.path.join(SAVE_DIR, f"image_{index:05d}.png") # Save path for the image
    bpy.context.scene.render.filepath = img_path # Set render output path
    bpy.ops.render.render(write_still=True) # Render the image and save it

    # Calculate Azimuth (alpha) and Elevation (beta) angles
    # Direction vector from camera to target (origin)
    direction = Vector((0, 0, 0)) - cam_loc

    # Azimuth: Angle in the XY plane from the positive X-axis. Ranges from -pi to pi.
    azimuth_rad = math.atan2(direction.y, direction.x)
    # Elevation: Angle from the XY plane towards the Z-axis. Ranges from -pi/2 to pi/2.
    elevation_rad = math.atan2(direction.z, math.sqrt(direction.x**2 + direction.y**2))

    # Convert radians to degrees for easier understanding in metadata
    azimuth_deg = math.degrees(azimuth_rad)
    elevation_deg = math.degrees(elevation_rad)

    # Normalize camera coordinates (x, y, z) to the range [-1, 1]
    # Divide by the maximum possible coordinate value (CAMERA_DISTANCE)
    norm_x = cam_loc.x / camera_distance_max
    norm_y = cam_loc.y / camera_distance_max
    norm_z = cam_loc.z / camera_distance_max

    # Normalize azimuth (alpha) to [-1, 1]
    # Azimuth ranges from -180 to 180 degrees.
    # (azimuth_deg + 180) maps it to [0, 360].
    # / 360 maps it to [0, 1].
    # * 2 - 1 maps it to [-1, 1].
    norm_azimuth = (azimuth_deg + 180) / 360 * 2 - 1

    # Normalize elevation (beta) to [-1, 1]
    # Elevation ranges from -90 to 90 degrees.
    # (elevation_deg + 90) maps it to [0, 180].
    # / 180 maps it to [0, 1].
    # * 2 - 1 maps it to [-1, 1].
    norm_elevation = (elevation_deg + 90) / 180 * 2 - 1

    # Write the metadata to the CSV file
    # Format: index,image_path,normalized_x,normalized_y,normalized_z,normalized_azimuth,normalized_elevation
    metadata_file.write(
        f"{index},{img_path},{norm_x:.4f},{norm_y:.4f},{norm_z:.4f},"
        f"{norm_azimuth:.4f},{norm_elevation:.4f}\n"
    )

# ========== MAIN EXECUTION FUNCTION ==========
def main():
    """
    Main function to set up the Blender scene, add objects,
    configure render settings, and render images for the dataset.
    """
    print("Starting Blender dataset generation...")
    clear_scene() # Start with a clean scene

    # Create the right-angle scalene triangle
    triangle, verts = create_triangle(TRIANGLE_SCALE_FACTOR)

    # Add colored spheres at the vertices of the triangle
    # Using accurate RGB colors for better distinction
    add_colored_sphere(verts[0], (1.0, 0.0, 0.0), "Red_Vertex_Sphere")     # Red (at (0, 4*scale, 0))
    add_colored_sphere(verts[1], (0.0, 1.0, 0.0), "Green_Vertex_Sphere")   # Green (at (0, 0, 0))
    add_colored_sphere(verts[2], (0.0, 0.0, 1.0), "Blue_Vertex_Sphere")    # Blue (at (3*scale, 0, 0))

    add_light() # Add lighting to the scene

    # Configure Blender render settings
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES' # Use Cycles render engine for high quality
    scene.cycles.samples = 64 # Number of samples for rendering (adjust for quality vs. speed)
    scene.cycles.use_denoising = False # Disable denoising for consistent output
    scene.render.resolution_x = 512 # Image width
    scene.render.resolution_y = 512 # Image height
    scene.render.image_settings.file_format = 'PNG' # Output image format

    # Open the metadata CSV file for writing
    metadata_path = os.path.join(SAVE_DIR, "metadata.csv")
    with open(metadata_path, "w") as f:
        # Write the header row for the CSV file, including normalized values
        f.write("index,image_path,norm_x,norm_y,norm_z,norm_azimuth,norm_elevation\n")
        
        # Loop to create, render, and clean up cameras for each sample
        for i in range(NUM_CAMERAS):
            if i % 1000 == 0: # Print progress every 1000 images
                print(f"Rendering image {i+1}/{NUM_CAMERAS}...")
            
            # Add a camera at a specific position on the sphere
            cam, cam_loc = add_camera_on_sphere(i, NUM_CAMERAS, CAMERA_DISTANCE)
            
            # Render the image from the current camera and save its metadata
            render_image(cam, i, cam_loc, f, CAMERA_DISTANCE)
            
            # Remove the camera object after rendering to keep the scene clean
            bpy.data.objects.remove(cam, do_unlink=True)

    print("✅ Dataset rendering complete!")
    print(f"Dataset saved to: {SAVE_DIR}")
    print(f"Metadata saved to: {metadata_path}")
    print(f"Total images rendered: {NUM_CAMERAS}")

# ========== SCRIPT ENTRY POINT ==========
if __name__ == "__main__":
    main()