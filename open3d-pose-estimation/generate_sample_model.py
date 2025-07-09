import open3d as o3d
import os
import argparse

def generate_simple_box(output_dir, filename, align_to_scene=None):
    print("[INFO] Generating Rectangular Object model...")
    width, height, depth = 0.06, 0.04, 0.008  # 60x40x8 mm
    box = o3d.geometry.TriangleMesh.create_box(width, height, depth)
    box.compute_vertex_normals()
    box.paint_uniform_color([0.2, 0.6, 0.8])

    if align_to_scene:
        print(f"[INFO] Aligning mesh to scene from: {align_to_scene}")
        scene = o3d.io.read_point_cloud(align_to_scene)
        if scene.is_empty():
            raise ValueError("Failed to load scene point cloud.")
        box.translate(scene.get_center() - box.get_center())

    model_pcd = box.sample_points_uniformly(number_of_points=2000)

    output_path = os.path.join(output_dir, filename)
    o3d.io.write_point_cloud(output_path, model_pcd)
    print(f"[INFO] Box point cloud saved to: {output_path}")
    o3d.visualization.draw_geometries([model_pcd], window_name="Box Point Cloud")

def main():
    parser = argparse.ArgumentParser(description="Generate 3D model representations using Open3D.")

    parser.add_argument("--output_dir", default="icp_data", help="Directory to save the output files")
    parser.add_argument("--output_file_name", default="sample_model.ply", help="Name of the output file")
    parser.add_argument("--scene", help="Optional RealSense scene PLY file for ground alignment")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    generate_simple_box(args.output_dir, args.output_file_name, align_to_scene=args.scene)

if __name__ == "__main__":
    main()
