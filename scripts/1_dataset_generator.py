import bpy
import os
import math
import random
import sys
import copy
import shutil
import subprocess
from mathutils import Vector

def ensure_dependencies():
    """Ensure PyYAML is installed in Blender's Python environment."""
    try:
        import yaml
    except ImportError:
        print("PyYAML not found. Installing...")
        python_exe = sys.executable
        subprocess.call([python_exe, "-m", "pip", "install", "pyyaml"])
        try:
            import yaml
            print("PyYAML installed successfully.")
        except ImportError:
            print("Failed to install PyYAML. Please install it manually in Blender's python.")
            sys.exit(1)

ensure_dependencies()
import yaml

# --- GLOBAL CONFIG LOADER ---
def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Load Config
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config/config.yaml")
# If running inside Blender text editor, hardcode path or ensure file is next to blend file
if not os.path.exists(CONFIG_PATH):
    # Fallback for testing if file not found relative to script
    CONFIG_PATH = "config.yaml" 

if os.path.exists(CONFIG_PATH):
    CFG = load_config(CONFIG_PATH)
else:
    print(f"ERROR: Configuration file not found at {CONFIG_PATH}")
    sys.exit(1)

def setup_hardware():
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    
    # Set Resolution
    scene.render.resolution_x = CFG['dataset']['resolution'][0]
    scene.render.resolution_y = CFG['dataset']['resolution'][1]
    scene.render.film_transparent = True  # Crucial for Compositing
    
    try:
        cprefs = bpy.context.preferences.addons['cycles'].preferences
        cprefs.get_devices()
        cprefs.compute_device_type = 'CUDA'
        for d in cprefs.devices:
            d.use = (d.type == 'CUDA')
        scene.cycles.device = 'GPU'
    except:
        print("Using CPU.")

def get_background_images(bg_dir):
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tga'}
    bg_images = []
    if not os.path.exists(bg_dir):
        print(f"Warning: Background directory not found: {bg_dir}")
        return []
    for file in os.listdir(bg_dir):
        if any(file.lower().endswith(ext) for ext in valid_extensions):
            bg_images.append(os.path.join(bg_dir, file))
    return bg_images

def create_vertex_color_material(obj):
    mat = bpy.data.materials.new(name="PLY_Material")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    node_attr = nodes.new(type='ShaderNodeAttribute')
    node_attr.attribute_name = "Col"
    node_bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    node_output = nodes.new(type='ShaderNodeOutputMaterial')
    links.new(node_attr.outputs['Color'], node_bsdf.inputs['Base Color'])
    links.new(node_bsdf.outputs['BSDF'], node_output.inputs['Surface'])
    obj.data.materials.append(mat) if not obj.data.materials else None

def set_camera_transform(cam, obj, max_factor):
    bbox_world = [obj.matrix_world @ Vector(v) for v in obj.bound_box]
    center = sum(bbox_world, Vector((0, 0, 0))) / 8
    radius = max((v - center).length for v in bbox_world)
    
    fov = cam.data.angle
    # Ensure object is visible even at closest zoom
    min_distance = (radius * 1.2) / math.sin(fov / 2)
    
    actual_distance = random.uniform(min_distance, min_distance * max_factor)
    
    phi = random.uniform(0, 2 * math.pi)
    theta = random.uniform(math.radians(20), math.radians(160))
    
    x = actual_distance * math.sin(theta) * math.cos(phi)
    y = actual_distance * math.sin(theta) * math.sin(phi)
    z = actual_distance * math.cos(theta)
    
    cam.location = center + Vector((x, y, z))

def setup_compositor_nodes(output_dir):
    """
    Sets up the compositing tree to:
    1. Resize background randomly.
    2. Blur object alpha (soft edges).
    3. Alpha Over object onto background.
    """
    scene = bpy.context.scene
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()
    
    # Enable Object Index for Mask
    bpy.context.view_layer.use_pass_object_index = True

    # --- NODES ---
    
    # 1. Inputs
    rl_node = tree.nodes.new('CompositorNodeRLayers')
    image_node = tree.nodes.new('CompositorNodeImage') # For Background
    
    # 2. Background Manipulation (Resize)
    transform_node = tree.nodes.new('CompositorNodeTransform')
    scale_node = tree.nodes.new('CompositorNodeScale') 
    scale_node.space = 'RENDER_SIZE'
    scale_node.inputs[1].default_value = 1 
    scale_node.inputs[2].default_value = 1 

    # 3. Object Manipulation (Blur Edges)
    blur_node = tree.nodes.new('CompositorNodeBlur')
    blur_node.filter_type = 'GAUSS'
    
    # --- FIX: Force integer type for blur size ---
    blur_radius = int(CFG['parameters']['edge_blur_radius'])
    blur_node.size_x = blur_radius
    blur_node.size_y = blur_radius
    
    set_alpha_node = tree.nodes.new('CompositorNodeSetAlpha')
    
    # 4. Mixing
    alpha_over_node = tree.nodes.new('CompositorNodeAlphaOver')
    alpha_over_node.use_premultiply = True
    
    # 5. Mask Generation
    id_mask_node = tree.nodes.new('CompositorNodeIDMask')
    id_mask_node.index = 1
    
    # 6. Outputs
    comp_output = tree.nodes.new('CompositorNodeComposite')
    
    file_output_img = tree.nodes.new('CompositorNodeOutputFile')
    file_output_img.base_path = os.path.join(output_dir, "images")
    file_output_img.format.file_format = 'PNG'
    
    # Output Mask (Standard binary mask)
    file_output_mask = tree.nodes.new('CompositorNodeOutputFile')
    file_output_mask.base_path = os.path.join(output_dir, "masks")
    file_output_mask.format.file_format = 'PNG'
    file_output_mask.format.color_mode = 'BW' 

    # --- LINKS ---
    
    # Background pipeline
    # Note: image_node needs an image loaded later in the loop to be valid
    tree.links.new(image_node.outputs['Image'], scale_node.inputs['Image'])
    tree.links.new(scale_node.outputs['Image'], transform_node.inputs['Image'])
    tree.links.new(transform_node.outputs['Image'], alpha_over_node.inputs[1]) # Background slot
    
    # Object Blur pipeline (Blurring the Alpha)
    tree.links.new(rl_node.outputs['Alpha'], blur_node.inputs['Image'])
    tree.links.new(rl_node.outputs['Image'], set_alpha_node.inputs['Image'])
    tree.links.new(blur_node.outputs['Image'], set_alpha_node.inputs['Alpha'])
    tree.links.new(set_alpha_node.outputs['Image'], alpha_over_node.inputs[2]) # Foreground slot
    
    # Output Image
    tree.links.new(alpha_over_node.outputs['Image'], comp_output.inputs['Image'])
    tree.links.new(alpha_over_node.outputs['Image'], file_output_img.inputs[0])
    
    # Output Mask
    tree.links.new(rl_node.outputs['IndexOB'], id_mask_node.inputs[0])
    tree.links.new(id_mask_node.outputs['Alpha'], file_output_mask.inputs[0])

    # Return dictionary of nodes we need to access later
    return {
        'image_node': image_node,
        'transform_node': transform_node,
        'img_output': file_output_img,
        'mask_output': file_output_mask
    }

def process_object(obj_path, bg_images, comp_nodes):
    print(f"Processing object: {os.path.basename(obj_path)}")
    
    # Import
    bpy.ops.import_mesh.ply(filepath=obj_path)
    obj = bpy.context.selected_objects[0]
    obj.pass_index = 1 # For Mask
    create_vertex_color_material(obj)
    
    # Center Object
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    obj.location = (0, 0, 0)

    # Setup Camera & Light
    if not bpy.context.scene.camera:
        bpy.ops.object.camera_add()
        cam = bpy.context.object
        bpy.context.scene.camera = cam
    else:
        cam = bpy.context.scene.camera
        
    # Track Constraint
    track = cam.constraints.new(type='TRACK_TO')
    track.target = obj
    track.track_axis = 'TRACK_NEGATIVE_Z'
    track.up_axis = 'UP_Y'

    # Sun Light
    lights = [o for o in bpy.context.scene.objects if o.type == 'LIGHT']
    if not lights:
        bpy.ops.object.light_add(type='SUN')
        sun = bpy.context.object
        sun.data.energy = 4
    else:
        sun = lights[0]

    obj_name = os.path.splitext(os.path.basename(obj_path))[0]
    
    img_count = CFG['dataset']['img_count']
    bg_scale_min = CFG['parameters']['bg_scale_range'][0]
    bg_scale_max = CFG['parameters']['bg_scale_range'][1]


    for i in range(img_count):
        # Randomize Object
        obj.rotation_euler = [random.uniform(0, 6.28) for _ in range(3)]
        sun.location = (random.uniform(-10, 10), random.uniform(-10, 10), 10)
        
        # Randomize Camera
        set_camera_transform(cam, obj, CFG['parameters']['max_distance_factor'])
        
        # Randomize Background
        if bg_images:
            bg_path = random.choice(bg_images)
            try:
                img = bpy.data.images.load(bg_path, check_existing=True)
                comp_nodes['image_node'].image = img
            except:
                print(f"Failed to load {bg_path}")
        
        # Randomize Background Transform
        # Random Scale
        scale = random.uniform(bg_scale_min, bg_scale_max)
        comp_nodes['transform_node'].inputs['Scale'].default_value = scale
        # Random Shift (To utilize the extra scaled area)
        # Shift X/Y in pixels roughly
        shift_range = 200 * (scale - 1.0) 
        comp_nodes['transform_node'].inputs['X'].default_value = random.uniform(-shift_range, shift_range)
        comp_nodes['transform_node'].inputs['Y'].default_value = random.uniform(-shift_range, shift_range)

        # Update Output Names
        # Format: objectname_0000
        comp_nodes['img_output'].file_slots[0].path = f"{obj_name}_{i:04d}_"
        comp_nodes['mask_output'].file_slots[0].path = f"{obj_name}_mask_{i:04d}_"

        print(f"Rendering {obj_name} frame {i+1}/{img_count}...")
        bpy.ops.render.render(write_still=False) # Compositor handles saving

    # Cleanup object after processing
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.ops.object.delete()

def run_pipeline():
    # Clear Scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
    setup_hardware()

    bg_images = get_background_images(CFG['dataset']['bg_dir'])

    for obj_path in CFG['objects']:
        if os.path.exists(obj_path):
            output_dir = CFG['dataset']['output_dir']
            
            object_name = os.path.basename(obj_path)

            object_folder_path = os.path.join(output_dir, object_name[:-4])
            os.makedirs(object_folder_path, exist_ok=True)
            shutil.copy(obj_path,os.path.join(object_folder_path, object_name))

            object_folder_path = os.path.join(object_folder_path, "augmented")

            os.makedirs(object_folder_path, exist_ok=True)
            os.makedirs(os.path.join(object_folder_path, "images"), exist_ok=True)
            os.makedirs(os.path.join(object_folder_path, "masks"), exist_ok=True)

            comp_nodes = setup_compositor_nodes(object_folder_path)
            
            
            process_object(obj_path, bg_images, comp_nodes)
        else:
            print(f"Error: Object file not found: {obj_path}")

if __name__ == "__main__":
    run_pipeline()
