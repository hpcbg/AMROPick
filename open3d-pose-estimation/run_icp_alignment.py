import open3d as o3d
import numpy as np
import argparse
import copy

def draw_registration_result(scene, model, transformation=None, window_name="Registration"):
    if transformation is not None:
        model = copy.deepcopy(model)
        model.transform(transformation)

    # # Add coordinate frame to show model orientation (default size = 1.0)
    # # You can adjust size and origin if needed
    # axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    # axis.transform(transformation)  # Apply the model's pose

    o3d.visualization.draw_geometries([scene, model], window_name=window_name)

def compute_fpfh(pcd, voxel_size):
    radius_normal = voxel_size * 2
    radius_feature = voxel_size * 5
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    return o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )

def run_alignment(model_path, scene_path, voxel_size, init_translation, init_rotation, visualize, skip_ransac, icp_threshold):
    print(f"[INFO] Loading model: {model_path}")
    model = o3d.io.read_point_cloud(model_path)
    print(f"[INFO] Loading scene: {scene_path}")
    scene = o3d.io.read_point_cloud(scene_path)

    initial_translation = scene.get_center() - model.get_center()
    T_init = np.eye(4)
    T_init[:3, 3] = initial_translation
    model.translate(initial_translation)

    if visualize:
        print("[INFO] Visualizing initial scene and model...")
        draw_registration_result(scene, model, window_name="Initial Scene and Model")

    if init_rotation:
        print(f"[INFO] Applying initial rotation (degrees): {init_rotation}")
        radians = np.radians(init_rotation)
        R = o3d.geometry.get_rotation_matrix_from_xyz(radians)
        model.rotate(R, center=model.get_center())

    if init_translation:
        print(f"[INFO] Applying initial translation: {init_translation}")
        model.translate(init_translation)

    if visualize:
        print("[INFO] Visualizing after model transformation...")
        draw_registration_result(scene, model, window_name="After Initial Transform")

    print("[INFO] Downsampling...")
    model_down = model.voxel_down_sample(voxel_size)
    scene_down = scene.voxel_down_sample(voxel_size)

    # if visualize:
    #     model_down.paint_uniform_color([1, 0, 0])  # red
    #     scene_down.paint_uniform_color([0, 0, 1])  # blue
    #     draw_registration_result(scene_down, model_down, window_name="Downsampled Clouds")

    if skip_ransac:
        print("[INFO] Skipping RANSAC. Using identity matrix for initial alignment.")
        init_transformation = np.identity(4)
    else:
        print("[INFO] Estimating FPFH features...")
        model_fpfh = compute_fpfh(model_down, voxel_size)
        scene_fpfh = compute_fpfh(scene_down, voxel_size)

        distance_threshold = voxel_size * 5.0
        print(f"[INFO] Running RANSAC (threshold = {distance_threshold:.4f})...")
        result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            model_down, scene_down, model_fpfh, scene_fpfh, mutual_filter=True,
            max_correspondence_distance=distance_threshold,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=4,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(2000000, 1000)
        )

        print("RANSAC transformation:\n", result_ransac.transformation)
        init_transformation = result_ransac.transformation

        if visualize:
            print("[INFO] Visualizing after RANSAC...")
            draw_registration_result(scene, model, init_transformation, window_name="After RANSAC")

    print(f"[INFO] Refining with ICP (threshold = {icp_threshold:.4f})...")
    scene.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=icp_threshold * 2, max_nn=30))
    model.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=icp_threshold * 2, max_nn=30))

    result_icp = o3d.pipelines.registration.registration_icp(
        model, scene, max_correspondence_distance=icp_threshold,
        init=init_transformation,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )

    print("ICP transformation:\n", result_icp.transformation)
    # model.transform(result_icp.transformation)

    print("[INFO] Visualizing final alignment...")
    draw_registration_result(scene, model, result_icp.transformation, window_name="Final Alignment: Model + Scene")

    T_total = result_icp.transformation @ T_init
    result_icp.transformation = T_total  # replace with the full transformation

    return result_icp


def main():
    parser = argparse.ArgumentParser(description="ICP Alignment using Open3D with optional RANSAC")
    parser.add_argument('--model', type=str, required=True, help='Path to model PLY file')
    parser.add_argument('--scene', type=str, required=True, help='Path to scene PLY file')
    parser.add_argument('--voxel', type=float, default=0.005, help='Voxel size for downsampling and features')

    parser.add_argument('--init_x', type=float, default=0.0, help='Initial X translation of model')
    parser.add_argument('--init_y', type=float, default=0.0, help='Initial Y translation of model')
    parser.add_argument('--init_z', type=float, default=0.0, help='Initial Z translation of model')

    parser.add_argument('--init_rx', type=float, default=0.0, help='Initial rotation around X axis (degrees)')
    parser.add_argument('--init_ry', type=float, default=0.0, help='Initial rotation around Y axis (degrees)')
    parser.add_argument('--init_rz', type=float, default=0.0, help='Initial rotation around Z axis (degrees)')

    parser.add_argument('--icp_threshold', type=float, default=0.03, help='ICP max correspondence distance')
    parser.add_argument('--skip_ransac', action='store_true', help='Skip RANSAC and go straight to ICP')
    parser.add_argument('--visualize', action='store_true', help='Enable intermediate visualizations')
    args = parser.parse_args()

    init_translation = [args.init_x, args.init_y, args.init_z]
    init_rotation = [args.init_rx, args.init_ry, args.init_rz]

    run_alignment(
        model_path=args.model,
        scene_path=args.scene,
        voxel_size=args.voxel,
        init_translation=init_translation,
        init_rotation=init_rotation,
        visualize=args.visualize,
        skip_ransac=args.skip_ransac,
        icp_threshold=args.icp_threshold
    )

if __name__ == "__main__":
    main()
