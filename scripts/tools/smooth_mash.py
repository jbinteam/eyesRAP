import open3d as o3d
import numpy as np

INPUT_FILE = "/home/oongking/iRAP/projects/eyesRAP/ply_model/emergency_bright.ply"
OUTPUT_FILE = "/home/oongking/iRAP/projects/eyesRAP/ply_model/emergency_bright_smooth.ply"

# only use with brighted mesh 
def make_smoother(input_path, output_path):
    mesh_in = o3d.io.read_triangle_mesh(input_path)
    mesh_in.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh_in])

    mesh_in.remove_duplicated_vertices()
    mesh_in.remove_degenerate_triangles()
    mesh_in.remove_non_manifold_edges()

    mesh_out = mesh_in.subdivide_midpoint(1)

    # mesh_out = mesh_in.filter_smooth_taubin(
    #     number_of_iterations=100
    # )

    print('filter with average with 5 iterations')
    mesh_out = mesh_in.filter_smooth_simple(number_of_iterations=5)
    mesh_out.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh_out])

    mesh_out.remove_non_manifold_edges()
    mesh_out.compute_vertex_normals()

    pcd_tree = o3d.geometry.KDTreeFlann(mesh_in)

    new_colors = []
    colors = np.asarray(mesh_in.vertex_colors)
    for v in mesh_out.vertices:
        _, idx, _ = pcd_tree.search_knn_vector_3d(v, 1)
        new_colors.append(colors[idx[0]])

    mesh_out.vertex_colors = o3d.utility.Vector3dVector(new_colors)
    mesh_in.translate([10,0,0])

    o3d.visualization.draw_geometries([mesh_out,mesh_in])

    print(f"Saving to {output_path}...")
    # o3d.io.write_triangle_mesh(output_path, mesh_out,write_ascii=True)

if __name__ == "__main__":
    make_smoother(INPUT_FILE, OUTPUT_FILE)
