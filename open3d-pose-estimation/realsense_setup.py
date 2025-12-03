import pyrealsense2 as rs
import numpy as np
from utils import load_config


def start_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 15)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 15)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    return pipeline, align, profile


def capture_frames(pipeline, align, profile, apply_filters=False):
    config = load_config()

    color_sensor = profile.get_device().first_color_sensor()

    color_sensor.set_option(rs.option.enable_auto_exposure, 1)
    color_sensor.set_option(rs.option.global_time_enabled, 1)
    color_sensor.set_option(rs.option.backlight_compensation, 1)
    color_sensor.set_option(rs.option.auto_exposure_priority, 1)

    # Manual color parameters (only take effect if auto-exposure is OFF)
    # color_sensor.set_option(rs.option.sharpness, config["camera"]["rgb"]["sharpness"])
    # color_sensor.set_option(rs.option.contrast, config["camera"]["rgb"]["contrast"])
    # color_sensor.set_option(rs.option.gamma, config["camera"]["rgb"]["gamma"])
    # color_sensor.set_option(rs.option.saturation, config["camera"]["rgb"]["saturation"])
    color_sensor.set_option(rs.option.enable_auto_white_balance, 1)
    
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_sensor.set_option(rs.option.emitter_enabled, 1)  # Laser ON
    depth_sensor.set_option(rs.option.enable_auto_exposure, 1)
   
    
    print("[INFO] Warming up sensor...")
    for _ in range(15):
        pipeline.wait_for_frames()

    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)

    depth = aligned_frames.get_depth_frame()
    color = aligned_frames.get_color_frame()

    if apply_filters:
        realsense_settings = load_config(path="amropick_realsense_settings.yaml")

        hole_filling = rs.hole_filling_filter()
        depth_to_disparity = rs.disparity_transform(True)
        disparity_to_depth = rs.disparity_transform(False)

        spatial = rs.spatial_filter()
        spatial.set_option(rs.option.filter_magnitude, realsense_settings["post_processing"]["spatial"]["magnitude"])  # 1–5
        spatial.set_option(rs.option.filter_smooth_alpha, realsense_settings["post_processing"]["spatial"]["smooth_alpha"])  # sharpening factor (0–1)
        spatial.set_option(rs.option.filter_smooth_delta, realsense_settings["post_processing"]["spatial"]["smooth_delta"])  # edge threshold
        
        temporal = rs.temporal_filter()
        temporal.set_option(rs.option.filter_smooth_alpha, realsense_settings["post_processing"]["temporal"]["smooth_alpha"])  # sharpening factor (0–1)
        temporal.set_option(rs.option.filter_smooth_delta, realsense_settings["post_processing"]["temporal"]["smooth_delta"])  # edge threshold

        hole_filling = rs.hole_filling_filter()

        depth = depth_to_disparity.process(depth)
        depth = spatial.process(depth)
        depth = temporal.process(depth)
        depth = hole_filling.process(depth)
        depth = disparity_to_depth.process(depth)

    colorizer = rs.colorizer()
    colorized_depth = np.asanyarray(colorizer.colorize(depth).get_data())

    return np.asanyarray(depth.get_data()), np.asanyarray(color.get_data()), colorized_depth

