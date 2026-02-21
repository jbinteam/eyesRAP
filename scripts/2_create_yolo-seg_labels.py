import os
import cv2
import glob
import numpy as np
from tqdm import tqdm
import yaml
import sys

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


def normalize_coordinates(contour, width, height):
    """Converts pixel coordinates to normalized (0-1) coordinates."""
    normalized_poly = []
    for point in contour:
        x, y = point[0]
        # YOLO format limits coordinates to [0, 1]
        normalized_poly.append(max(0, min(1, x / width)))
        normalized_poly.append(max(0, min(1, y / height)))
    return normalized_poly

def process_masks_for_object(object_name, mask_dir, label_dir, class_id):
    """Processes all masks in a specific directory."""
    
    # Create labels folder if it doesn't exist
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
        
    mask_files = glob.glob(os.path.join(mask_dir, "*.png"))
    
    print(f"   -> Processing '{object_name}' (ID: {class_id}) | Found {len(mask_files)} masks.")

    for mask_file in tqdm(mask_files, leave=False):
        # Read Mask
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        if mask is None: continue

        height, width = mask.shape
        _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        yolo_lines = []
        for contour in contours:
            if cv2.contourArea(contour) < 50: continue # Filter noise

            # Simplify polygon (lower epsilon = more detail)
            epsilon = 0.005 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Normalize
            poly = normalize_coordinates(approx, width, height)
            
            # Format: <class_id> <x1> <y1> <x2> <y2> ...
            line_content = f"{class_id} " + " ".join(map(str, poly))
            yolo_lines.append(line_content)

        # Output Filename: match the image name (remove _mask suffix)
        # e.g. emergency_mask_0000.png -> emergency_0000.txt
        filename = os.path.basename(mask_file)
        if "_mask_" in filename:
            txt_name = filename.replace("_mask_", "_")
        else:
            txt_name = filename
        
        txt_name = os.path.splitext(txt_name)[0] + ".txt"
        
        # Save Label
        if yolo_lines:
            with open(os.path.join(label_dir, txt_name), "w") as f:
                f.write("\n".join(yolo_lines))

def main():
    print("--- 🔍 Scanning for datasets in ply_model ---")
    
    ROOT_DIR = CFG['dataset']['output_dir']
    # 1. Discover Object Folders
    # We look for any folder inside ROOT_DIR that contains "augmented/masks"
    object_folders = []

    if not os.path.exists(ROOT_DIR):
        print(f"Error: Directory not found: {ROOT_DIR}")
        return

    # Walk through immediate subdirectories of ROOT_DIR
    for item in os.listdir(ROOT_DIR):
        item_path = os.path.join(ROOT_DIR, item)
        
        if os.path.isdir(item_path):
            mask_path = os.path.join(item_path, "augmented", "masks")
            if os.path.exists(mask_path):
                object_folders.append(item)

    object_folders.sort() # Ensure consistent ID assignment
    
    if not object_folders:
        print("No 'augmented/masks' folders found. Check your directory structure.")
        return

    print(f"Found {len(object_folders)} objects: {object_folders}")

    # 2. Create Class Mapping (dataset.yaml info)
    print("\n--- 🏷️  Assigning Class IDs ---")
    class_map = {name: idx for idx, name in enumerate(object_folders)}
    
    # Save a classes.txt for reference
    with open("config/classes.txt", "w") as f:
        for name, idx in class_map.items():
            f.write(f"{idx}: {name}\n")
            print(f"ID {idx}: {name}")
    print("(Saved mapping to classes.txt)")

    # 3. Process Each Object
    print("\n--- 🚀 Generating Labels ---")
    for obj_name in object_folders:
        obj_dir = os.path.join(ROOT_DIR, obj_name)
        mask_dir = os.path.join(obj_dir, "augmented", "masks")
        label_dir = os.path.join(obj_dir, "augmented", "labels")
        
        process_masks_for_object(obj_name, mask_dir, label_dir, class_map[obj_name])

    print("\n✅ Done! Labels created in 'augmented/labels' for each object.")

if __name__ == "__main__":
    main()