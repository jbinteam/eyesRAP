import cv2
import os
import sys
import glob
import yaml
import random
import numpy as np
from PIL import Image  # <--- Added PIL import

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

COLORS = [
    (0, 255, 0),    # Green
    (0, 0, 255),    # Red
    (255, 0, 0),    # Blue
    (0, 255, 255),  # Yellow
    (255, 0, 255)   # Magenta
]

def load_classes():
    """Try to load class names from classes.txt if it exists."""
    class_names = {}
    if os.path.exists("config/classes.txt"):
        with open("config/classes.txt", "r") as f:
            for line in f:
                parts = line.strip().split(":")
                if len(parts) == 2:
                    class_names[int(parts[0])] = parts[1].strip()
    return class_names

def load_image_clean(path):
    """
    Loads an image using PIL instead of OpenCV to avoid libpng warnings.
    Converts PIL (RGB) -> OpenCV (BGR) format.
    """
    try:
        # PIL is tolerant of the 'incorrect sRGB profile'
        pil_img = Image.open(path).convert('RGB')
        
        # Convert PIL image to Numpy array
        open_cv_image = np.array(pil_img)
        
        # Convert RGB to BGR (OpenCV uses BGR)
        open_cv_image = open_cv_image[:, :, ::-1].copy()
        
        return open_cv_image
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def draw_yolo_label(image, label_path, class_map):
    h, w = image.shape[:2]
    
    with open(label_path, "r") as f:
        lines = f.readlines()
        
    for line in lines:
        data = list(map(float, line.strip().split()))
        class_id = int(data[0])
        
        # YOLO format: class_id x1 y1 x2 y2 ... xn yn (Normalized 0-1)
        coords = data[1:]
        points = []
        
        for i in range(0, len(coords), 2):
            x = int(coords[i] * w)
            y = int(coords[i+1] * h)
            points.append([x, y])
            
        points = np.array(points, np.int32)
        points = points.reshape((-1, 1, 2))
        
        # 1. Draw Polygon Outline
        color = COLORS[class_id % len(COLORS)]
        cv2.polylines(image, [points], isClosed=True, color=color, thickness=2)
        
        # 2. Draw Filled Polygon (Semi-transparent)
        overlay = image.copy()
        cv2.fillPoly(overlay, [points], color)
        cv2.addWeighted(overlay, 0.4, image, 0.6, 0, image)
        
        # 3. Draw Text Label
        label_text = class_map.get(class_id, f"Class {class_id}")
        
        # Find the top-most point to place text safely
        if len(points) > 0:
            top_point = min(points, key=lambda p: p[0][1])[0]
            # Ensure text doesn't go off-screen top
            text_y = max(20, top_point[1] - 10)
            
            cv2.putText(image, label_text, (top_point[0], text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return image

def main():
    print("--- 🕵️ YOLO Label Visualizer ---")
    print("Controls: [Space] Next Image | [Q] Quit")

    ROOT_DIR = CFG['dataset']['output_dir']

    class_map = load_classes()
    image_files = []

    # 1. Find all images recursively
    for root, dirs, files in os.walk(ROOT_DIR):
        for file in files:
            if file.endswith(".png") and "masks" not in root:
                if "images" in root:
                    image_files.append(os.path.join(root, file))

    if not image_files:
        print(f"No images found in {ROOT_DIR}")
        return

    print(f"Found {len(image_files)} images. shuffling...")
    random.shuffle(image_files)

    for img_path in image_files:
        label_path = img_path.replace("images", "labels").replace(".png", ".txt")
        
        if not os.path.exists(label_path):
            continue
            
        # --- CHANGED: Use the helper function instead of cv2.imread ---
        image = load_image_clean(img_path)
        
        if image is None: continue
        
        visualized_img = draw_yolo_label(image, label_path, class_map)
        
        cv2.imshow("Label Check (Press Space for Next, Q to Quit)", visualized_img)
        
        key = cv2.waitKey(0)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()