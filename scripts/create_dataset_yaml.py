import os
import yaml

# --- CONFIGURATION ---
# The root folder where your objects are stored
# Script expects structure: ROOT_DIR / object_name / augmented / {images, labels}
ROOT_DIR = "/home/jb/dataset_generator/ply_model"


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
    yaml_path = "custom_data.yaml"
    with open(yaml_path, 'w') as f:
        # default_flow_style=None lets it choose the best layout (lists vs blocks)
        yaml.dump(data_yaml, f, sort_keys=False, default_flow_style=None)
    
    print(f"Saved configuration to {yaml_path}")
    return yaml_path

if __name__ == "__main__":
    dataset_yaml=create_dataset_yaml(ROOT_DIR)