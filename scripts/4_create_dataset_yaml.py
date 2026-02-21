import os
import sys
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
    # --- CONFIGURATION ---
    # The root folder containing your object folders
    # Structure expected: ROOT_DIR / <object_name> / augmented / masks
    # ROOT_DIR = CFG['dataset']['output_dir']
else:
    print(f"ERROR: Configuration file not found at {CONFIG_PATH}")
    sys.exit(1)


def create_dataset_yaml(root_dir):
    """
    Scans the directory to find all object classes and creates a data.yaml
    that YOLOv8 can read. Now outputs CLEAN YAML (no &id anchors).
    """
    print("--- 🛠️  Building data.yaml ---")
    
    # 1. Identify Classes
    classes = []
    for item in os.listdir(root_dir):
        if os.path.isdir(os.path.join(root_dir, item)):
            if os.path.exists(os.path.join(root_dir, item, "augmented", "images")):
                classes.append(item)
    
    classes.sort()
    
    if not classes:
        print("Error: No valid object datasets found in", root_dir)
        return None

    print(f"Found {len(classes)} classes: {classes}")

    # 2. Build Paths List
    train_paths = []
    for cls in classes:
        img_path = os.path.join(root_dir, cls, "augmented", "images")
        train_paths.append(img_path)

    # 3. Create Dictionary
    # FIX: Use .copy() for validation so YAML doesn't create an anchor (&id)
    data_yaml = {
        'path': root_dir, 
        'train': train_paths, 
        'val': train_paths.copy(),  # <--- .copy() fixes the &id issue!
        'names': {i: name for i, name in enumerate(classes)}
    }

    # 4. Save to file
    yaml_path = "config/custom_data.yaml"
    with open(yaml_path, 'w') as f:
        # default_flow_style=None lets it choose the best layout (lists vs blocks)
        yaml.dump(data_yaml, f, sort_keys=False, default_flow_style=None)
    
    print(f"Saved configuration to {yaml_path}")
    return yaml_path

if __name__ == "__main__":
    ROOT_DIR = CFG['dataset']['output_dir']
    dataset_yaml=create_dataset_yaml(ROOT_DIR)