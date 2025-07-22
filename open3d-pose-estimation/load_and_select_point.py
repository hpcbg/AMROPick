import open3d as o3d
import numpy as np


def manual_grasp_point_select(mesh_path, visualize_all = False, resampling = 5000, diameter_rescale = 100):
    # Load mesh
    
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    # Make normals if not there - needed for back point removal
    mesh.compute_vertex_normals()

    # Sample points using poisson. Easier to pick points - 5000 just as a baseline
    pcd = mesh.sample_points_poisson_disk(resampling)

    if visualize_all:
        o3d.visualization.draw_geometries([pcd], "Resampled mesh into point cloud")

    # get a diameter of what the camera sees - something like FOV based on the bounds of the mesh
    diameter = np.linalg.norm(
        np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound())
    )

    # get position of the camera at x = 0, y = 0, z = diameter
    camera = [0, 0, diameter]
    #scale if necessary
    diameter_scaled = diameter * diameter_rescale

    #get only points that are visible from the camera's position and in a fov
    print("Get all points that are visible from given view point")
    _, pt_map = pcd.hidden_point_removal(camera, diameter_scaled)

    pcd_onesided = pcd.select_by_index(pt_map)

    if visualize_all:
        o3d.visualization.draw_geometries([pcd_onesided], "One sided point cloud only see from camera")

    # Visualize and pick points on the point cloud using the built-in point picker shift + left click
    print("Pick points by shift + left click, then close the window")

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd_onesided)
    vis.run()

    picked_indices = vis.get_picked_points()
    vis.destroy_window()

    # get the point for grasping and the normal at that point might be useful for grasping
    points = np.asarray(pcd_onesided.points)
    picked_coords = points[picked_indices]
    normals = np.asarray(pcd_onesided.normals)
    picked_normals = normals[picked_indices]


    return picked_coords, picked_normals


if __name__ == "__main__":

    # model_path = "..\pose-estimation-open3d\data\3D-models\Plate1.stl"
    model_path = "data/3D-models/Plate1.stl"

    grasp_point, grasp_normal = manual_grasp_point_select(model_path)

    for pt,norm in zip(grasp_point,grasp_normal):
        print(f"Point:{pt}, Normal: {norm}")