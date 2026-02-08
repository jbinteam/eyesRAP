Dataset Generator for Object Detection
This tool automates the creation of synthetic datasets for computer vision tasks (e.g., Object Detection, Instance Segmentation). It takes 3D scanned objects (.ply) created with RealityScan, superimposes them onto random background images from the VOC2012 dataset, and renders them using Blender 3.6 with varied augmentations.

Pipeline Overview
Input Object: High-quality 3D scans (.ply) generated via the RealityScan app.

Backgrounds: Real-world images from the VOC2012 dataset.

Augmentation Engine: Blender 3.6 (Cycles) script that randomizes:

3D Object Rotation (X, Y, Z)

Camera Distance & Angle

Lighting Direction (Sun Position)

Background Scaling & Position

Edge Blurring (Simulating depth of field/anti-aliasing)

Output:

RGB Images: Photorealistic composites of the object on random backgrounds.

Binary Masks: Perfect segmentation masks for the object.

Prerequisites
Blender 3.6 LTS (Tested on 3.6, likely works on 3.x+)

Ensure Blender is added to your system PATH or know the path to the executable.

Python Dependencies:

PyYAML (The script attempts to install this automatically within Blender's Python environment).

Data:

3D Models: .ply files from RealityScan.

Backgrounds: Pascal VOC 2012 Dataset (specifically the JPEGImages folder).

Installation & Setup
Clone/Download this repository.

Prepare your Data:

Place your .ply files in a known directory.

Extract the VOC2012 dataset and locate the JPEGImages folder.

Configure the Project:

Open config.yaml in a text editor.

Update the paths to match your system.

config.yaml Example:

YAML

dataset:
  output_dir: "/path/to/output/dataset"
  bg_dir: "/path/to/VOC2012/JPEGImages"
  img_count: 50         # Images per object
  resolution: [640, 480] # [Width, Height]

parameters:
  max_distance_factor: 2.5  # 1.0 = fit frame, 2.5 = zoomed out
  edge_blur_radius: 2       # Softens object edges (pixels)
  bg_scale_range: [1.0, 2.0] # Random background scaling

objects:
  - "/path/to/model1.ply"
  - "/path/to/model2.ply"
Usage
Run the generation script using Blender's background mode (-b) and python execution flag (-P).

Command Line:

Bash

blender -b -P generator.py
Breakdown:

blender: Calls the Blender executable.

-b: Runs in background mode (no UI window).

-P generator.py: Executes the python script.

Output Structure
The script will create the following directory structure in your defined output_dir:

Plaintext

output_dir/
├── images/
│   ├── object_name_0000.png
│   ├── object_name_0001.png
│   └── ...
└── masks/
    ├── object_name_mask_0000.png
    ├── object_name_mask_0001.png
    └── ...
Images: The augmented RGB image.

Masks: Black and white binary mask (White = Object, Black = Background).

Troubleshooting
HIP hipInit: Invalid device:

If you see this on an NVIDIA or CPU setup, you can ignore it. It is an AMD driver check.

ModuleNotFoundError: No module named 'yaml':

The script tries to auto-install PyYAML. If it fails, run this command manually using Blender's bundled Python:

Bash

/path/to/blender/3.6/python/bin/python3.10 -m pip install pyyaml
Pink/Purple Background:

This usually means the background image path is incorrect or the VOC folder is empty. Check bg_dir in config.yaml.