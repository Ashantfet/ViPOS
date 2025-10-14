import bpy
import os
import math
import random
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view
from datetime import datetime

# ==========================================================
# CONFIGURATION
# ==========================================================
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")
SAVE_DIR = f"/scratch/kalidas_1/project/tmp/apriltag_6dof_dataset_rendered"
APRILTAG_PATH = "/scratch/kalidas_1/project/tags/apriltag_36h11_00001.png"

os.makedirs(os.path.join(SAVE_DIR, "images"), exist_ok=True)

NUM_CAMERAS = 10000       # Number of views to render
CAMERA_DISTANCE = 15.0       # Distance of camera from tag (lower = closer)
PLANE_SIZE = 5.0             # Physical size of AprilTag plane
CAM_LENS = 10              # Camera lens focal length in mm

MIN_ELEV_DEG = 15            # Avoid grazing views (min elevation)
MAX_ELEV_DEG = 75            # Max elevation above plane
MIN_BBOX_AREA = 0.01         # Minimum visible tag area fraction (of image)
MAX_TRIES = 40               # Max tries to find valid camera position


# ==========================================================
# SCENE CLEANUP
# ==========================================================
def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    for coll in (bpy.data.meshes, bpy.data.lights, bpy.data.cameras, bpy.data.materials, bpy.data.images):
        for b in coll:
            try:
                coll.remove(b, do_unlink=True)
            except:
                pass

# ==========================================================
# APRILTAG PLANE CREATION
# ==========================================================
def create_apriltag_plane(size, texture_path):
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0, 0, 0))
    plane = bpy.context.active_object
    plane.name = "AprilTagPlane"

    mat = bpy.data.materials.new("AprilTagMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for n in nodes:
        nodes.remove(n)

    tex_node = nodes.new("ShaderNodeTexImage")
    tex_node.image = bpy.data.images.load(texture_path)
    tex_node.interpolation = 'Closest'

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = 0.5
    if "Specular" in bsdf.inputs:
        bsdf.inputs["Specular"].default_value = 0.2
    elif "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.2


    output = nodes.new("ShaderNodeOutputMaterial")
    links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    plane.data.materials.append(mat)
    return plane

# ==========================================================
# LIGHTS
# ==========================================================
def add_lights():
    bpy.ops.object.light_add(type='SUN', location=(5, 5, 10))
    bpy.context.active_object.data.energy = 6
    bpy.ops.object.light_add(type='SUN', location=(-5, -5, 8))
    bpy.context.active_object.data.energy = 4

# ==========================================================
# CAMERA UTILITIES
# ==========================================================
def ensure_camera_facing_tag(cam, target):
    direction_to_tag = (target - cam.location).normalized()
    cam_forward = -cam.matrix_world.to_quaternion() @ Vector((0, 0, 1))
    angle = cam_forward.angle(direction_to_tag, 0.0)
    if angle > math.radians(60):
        rot_quat = direction_to_tag.to_track_quat('-Z', 'Y')
        cam.rotation_euler = rot_quat.to_euler()

# Check visibility and projected area of tag
def tag_visible(scene, cam, obj, min_area_frac=MIN_BBOX_AREA, border_margin=0.05):
    """
    Check if tag is fully visible within frame and large enough.
    border_margin = fraction of image border to keep free (e.g. 0.05 = 5%)
    """
    corners_world = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    proj = [world_to_camera_view(scene, cam, co) for co in corners_world]

    xs = [p.x for p in proj]
    ys = [p.y for p in proj]
    zs = [p.z for p in proj]

    # Reject if any corner is behind the camera
    if any(z <= 0.0 for z in zs):
        return False

    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    w, h = maxx - minx, maxy - miny
    area = w * h

    # Require sufficient projected area
    if area < min_area_frac:
        return False

    # Require the entire tag to fit well inside the frame (with safety margin)
    if (
        minx < border_margin
        or maxx > 1.0 - border_margin
        or miny < border_margin
        or maxy > 1.0 - border_margin
    ):
        return False

    return True


# ==========================================================
# RANDOMIZED CAMERA SAMPLING (6DoF)
# ==========================================================
def add_random_camera(index, radius, target=Vector((0, 0, 0)),
                      min_elev_deg=MIN_ELEV_DEG, max_elev_deg=MAX_ELEV_DEG):
    az = random.uniform(0.0, 2.0 * math.pi)
    min_phi = math.radians(90.0 - max_elev_deg)
    max_phi = math.radians(90.0 - min_elev_deg)
    phi = random.uniform(min_phi, max_phi)

    x = radius * math.sin(phi) * math.cos(az) + random.uniform(-1.0, 1.0)
    y = radius * math.sin(phi) * math.sin(az) + random.uniform(-1.0, 1.0)
    z = radius * math.cos(phi) + random.uniform(-1.0, 1.0)

    bpy.ops.object.camera_add(location=(x, y, z))
    cam = bpy.context.active_object
    cam.data.lens = CAM_LENS
    cam.rotation_mode = 'XYZ'

    # Randomize target a bit so all look directions aren't identical
    perturbed_target = Vector((
        target.x + random.uniform(-0.2, 0.2),
        target.y + random.uniform(-0.2, 0.2),
        target.z + random.uniform(-0.2, 0.2)
    ))

    direction = (perturbed_target - cam.location).normalized()
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam.rotation_euler = rot_quat.to_euler()

    # Now apply small random camera perturbations
    cam.rotation_euler.x += math.radians(random.uniform(-10, 10))  # pitch
    cam.rotation_euler.y += math.radians(random.uniform(-10, 10))  # roll
    cam.rotation_euler.z += math.radians(random.uniform(-30, 30))  # yaw

    # Clamp distance
    dist = cam.location.length
    if dist > radius * 1.05 or dist < radius * 0.95:
        cam.location = cam.location.normalized() * radius

    return cam, cam.location.copy()


# ==========================================================
# NORMALIZATION UTILITIES
# ==========================================================
def normalize_value(value, min_val, max_val):
    return ((value - min_val) / (max_val - min_val)) * 2 - 1

def normalize_pose(cam_loc, cam_rot):
    norm_x = cam_loc.x / CAMERA_DISTANCE
    norm_y = cam_loc.y / CAMERA_DISTANCE
    norm_z = cam_loc.z / CAMERA_DISTANCE
    roll = math.degrees(cam_rot[0])
    pitch = math.degrees(cam_rot[1])
    yaw = math.degrees(cam_rot[2])
    norm_roll = normalize_value(roll, -180, 180)
    norm_pitch = normalize_value(pitch, -180, 180)
    norm_yaw = normalize_value(yaw, -180, 180)
    return (norm_x, norm_y, norm_z, norm_roll, norm_pitch, norm_yaw)

# ==========================================================
# RENDER FUNCTION — writes to two CSVs
# ==========================================================
def render_image(cam, i, f_raw, f_norm):
    bpy.context.scene.camera = cam
    img_path = os.path.join(SAVE_DIR, "images", f"image_{i:05d}.png")
    bpy.context.scene.render.filepath = img_path
    bpy.ops.render.render(write_still=True)

    # --- Raw pose ---
    tx, ty, tz = cam.location
    roll, pitch, yaw = [math.degrees(a) for a in cam.rotation_euler]

    # --- Normalized pose ---
    nx, ny, nz, nroll, npitch, nyaw = normalize_pose(cam.location, cam.rotation_euler)

    # --- Write to both CSVs ---
    f_raw.write(f"{i},{img_path},{tx:.4f},{ty:.4f},{tz:.4f},{roll:.2f},{pitch:.2f},{yaw:.2f}\n")
    f_norm.write(f"{i},{img_path},{nx:.4f},{ny:.4f},{nz:.4f},{nroll:.4f},{npitch:.4f},{nyaw:.4f}\n")


# ==========================================================
# MAIN LOOP — handles two CSVs
# ==========================================================
def main():
    clear_scene()
    plane = create_apriltag_plane(PLANE_SIZE, APRILTAG_PATH)
    add_lights()

    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 32
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.image_settings.file_format = 'PNG'

    # --- Two output CSVs ---
    raw_csv = os.path.join(SAVE_DIR, "metadata_raw.csv")
    norm_csv = os.path.join(SAVE_DIR, "metadata_normalized.csv")

    with open(raw_csv, "w") as f_raw, open(norm_csv, "w") as f_norm:
        f_raw.write("index,path,tx,ty,tz,roll,pitch,yaw\n")
        f_norm.write("index,path,norm_x,norm_y,norm_z,norm_roll,norm_pitch,norm_yaw\n")

        for i in range(NUM_CAMERAS):
            if i % 100 == 0:
                print(f"Rendering {i}/{NUM_CAMERAS} ...")

            attempt = 0
            success = False
            while attempt < MAX_TRIES:
                attempt += 1
                cam, loc = add_random_camera(i, CAMERA_DISTANCE)
                if tag_visible(scene, cam, plane):
                    render_image(cam, i, f_raw, f_norm)
                    bpy.data.objects.remove(cam, do_unlink=True)
                    success = True
                    break
                bpy.data.objects.remove(cam, do_unlink=True)

            if not success:
                print(f"⚠️ Skipping {i}, could not find visible view after {MAX_TRIES} attempts.")

    print(f"✅ Done! Saved raw → {raw_csv}")
    print(f"✅ Done! Saved normalized → {norm_csv}")

 

# ==========================================================
# RUN SCRIPT
# ==========================================================
if __name__ == "__main__":
    main()
