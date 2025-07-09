import open3d as o3d
import numpy as np
import argparse
import os

def pick_points(pcd):
    """
    1) Pick a point [shift + left click]
    2) Press [shift + right click] to undo point picking
    3) After picking points, press 'Q' to close the window
    """
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()

    return vis.get_picked_points()


def RemoveNoiseStatistical(pcd, nb_neighbors=20, std_ratio=2.0):
    """ remove point clouds noise using statitical noise removal method

    Args:
        pc (ndarray): N x 3 point clouds
        nb_neighbors (int, optional): Defaults to 20.
        std_ratio (float, optional): Defaults to 2.0.

    Returns:
        [ndarray]: N x 3 point clouds
    """
    pcd_cleaned, ind = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio)

    return pcd_cleaned

def PlaneRegression(pcd, threshold=0.01, init_n=3, iter=1000):
    """ plane regression using ransac

    Args:
        points (ndarray): N x3 point clouds
        threshold (float, optional): distance threshold. Defaults to 0.003.
        init_n (int, optional): Number of initial points to be considered inliers in each iteration
        iter (int, optional): number of iteration. Defaults to 1000.

    Returns:
        [ndarray, List]: 4 x 1 plane equation weights, List of plane point index
    """

    w, index = pcd.segment_plane(
        threshold, init_n, iter)

    return w, index


try_remove_plane = False
manual_select = True
downsample = False

model_path = "../../pose-estimation-open3d/icp_data/sample_model.ply"

scene_path = "../../pose-estimation-open3d/icp_data/realsense_scene_2.ply"



print(f"[INFO] Loading model: {model_path}")
model = o3d.io.read_point_cloud(model_path)
print(f"[INFO] Loading scene: {scene_path}")
scene = o3d.io.read_point_cloud(scene_path)

model.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
scene.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

print(len(model.points))

o3d.visualization.draw_geometries([scene, model], window_name="Initial Scene and Model")

# Statistical noise removal - if there are any little floating points this can remove them
scene = RemoveNoiseStatistical(scene)

o3d.visualization.draw_geometries([scene, model], window_name="Cleaned Scene and Model")

if try_remove_plane:
    # Try to find a plane get the indices of the points that contain it
    _ , plane_indices = PlaneRegression(scene, threshold= 0.0001)

    # Create a boolean mask: True means keep the point
    mask = np.ones(len(scene.points), dtype=bool)
    mask[plane_indices] = False


    # Apply the mask to all attributes of the point cloud
    scene.points = o3d.utility.Vector3dVector(np.asarray(scene.points)[mask])

    # Keep colors/normals if they exist
    if scene.has_colors():
        scene.colors = o3d.utility.Vector3dVector(np.asarray(scene.colors)[mask])
    if scene.has_normals():
        scene.normals = o3d.utility.Vector3dVector(np.asarray(scene.normals)[mask])

    o3d.visualization.draw_geometries([scene], window_name="Removed Scene and Model")


initial_transform = np.identity(4)

if manual_select:
    picked_point_index = pick_points(scene)
    picked_points = np.asarray(scene.points)[picked_point_index[0]]

    model_centroid = model.get_center()
    translation = picked_points - model_centroid
    initial_transform[:3, 3] = translation




# # RANSAC does not work - too flat to get the initial transforms for ICP

# radius_feature = 1
# model_fpfh = o3d.pipelines.registration.compute_fpfh_feature(model, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
# scene_fpfh = o3d.pipelines.registration.compute_fpfh_feature(scene, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

# print("Number of features source:", len(model_fpfh.data[0]))
# print("Number of features target:", len(scene_fpfh.data[0]))


# # distance_threshold = 5 * 1.5

# # result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
# #         model, scene, model_fpfh, scene_fpfh, True,
# #         distance_threshold,
# #         o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
# #         4,  # RANSAC iterations per validation
# #         [
# #             o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
# #             o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
# #         ],
# #         o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))






if downsample:

    downsample_size = 0.005
    model = model.voxel_down_sample(downsample_size)
    scene = scene.voxel_down_sample(downsample_size)



# robust ICP

# Threshold make it bigger to try to move the model farther but can end up with incorrect positions if the outlier points are too many
threshold = 1

# Criterios for ending the ICP algorithm. Either relative closeness/fitness between the two point clouds, minimum RMSE between them or number of iterations to run
criteria = o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=0.000001,
                                       relative_rmse=0.0000001,
                                       max_iteration=100)

# Robust kernel used as a loss to remove outliers 
loss = o3d.pipelines.registration.TukeyLoss(k=0.1)
p2l = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
# Point to Point ICP - does not work as good as PPoint to Plane for the use case
p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()

reg_p2l = o3d.pipelines.registration.registration_icp(model, scene, threshold, initial_transform, p2l, criteria)
model.transform(reg_p2l.transformation)

print("ICP transformation:\n", reg_p2l.transformation)

o3d.visualization.draw_geometries([scene, model], window_name="Initial Scene and Model")