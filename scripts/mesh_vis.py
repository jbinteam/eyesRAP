import open3d as o3d
import numpy as np
import copy

INPUT_FILE = "/home/jb/dataset_generator/ply_model/emergency/emergency.ply"
OUTPUT_FILE = "emergency_bright.ply"

def make_brighter(input_path, output_path):
    print(f"Loading {input_path}...")
    mesh = o3d.io.read_triangle_mesh(input_path)

    if not mesh.has_vertex_colors():
        print("Error: No colors found!")
        return

    # 1. Get Colors
    colors = np.asarray(mesh.vertex_colors)
    max_val = np.max(colors)
    print(f"Original Max Brightness: {max_val:.4f}")

    # 2. AUTO-EXPOSURE (Normalize)
    # If the brightest pixel is 0.1, we multiply everything by 10.0
    if max_val > 0:
        scale_factor = 1.0 / max_val
        print(f"Scaling brightness by factor: {scale_factor:.2f}x")
        colors = colors * scale_factor
    
    # 3. GAMMA CORRECTION (Linear -> sRGB)
    # Standard monitor correction
    print("Applying Gamma Correction...")
    colors = np.power(colors, 1.0 / 2.2)

    # 4. SATURATION BOOST (Optional)
    # Sometimes normalizing makes colors look washed out. 
    # This simple trick boosts saturation slightly.
    # (Uncomment below line if colors look grey/pale)
    # colors = colors * 1.2 

    # Clip to ensure valid range (0.0 - 1.0)
    colors = np.clip(colors, 0.0, 1.0)

    # 5. Apply & Save
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    
    if not mesh.has_vertex_normals():
        print("Computing Normals...")
        mesh.compute_vertex_normals()

    print(f"Saving to {output_path}...")
#    o3d.io.write_triangle_mesh(output_path, mesh, write_ascii=True)
    
    print("Visualizing...")
    o3d.visualization.draw_geometries([mesh], window_name="Brighter PLY")

if __name__ == "__main__":
    make_brighter(INPUT_FILE, OUTPUT_FILE)
