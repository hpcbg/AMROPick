import pyrealsense2 as rs

def setup_pipeline():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 15)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 15)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    # Warm-up
    for _ in range(15):
        pipeline.wait_for_frames()

    return pipeline, align, profile