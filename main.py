from pathlib import Path
import argparse
import cv2
import numpy as np
import shutil
import process
import visualizer


# -------------------- USER SETTINGS --------------------
INPUT_VIDEO = "data/input/sac_2.mp4"
OUTPUT_DIR = "data/output"

# None keeps the original video resolution. 1280 is a reasonable starting point.
PROCESS_WIDTH = None
FRAME_GAP = 8

MAX_FEATURES = 2000
QUALITY_LEVEL = 0.005
MIN_DISTANCE = 5
FOCAL_MULTIPLIER = 1.2

MIN_TRACKED_POINTS = 30
MIN_ESSENTIAL_INLIERS = 20
MIN_POSE_VALID_POINTS = 15

DENSE_GRID_STEP = 3
FB_THRESHOLD = 1.5

# Each panel is this size; final analysis video is 2x2 panels.
PANEL_WIDTH = 1280
PANEL_HEIGHT = 720
DISPLAY = False
OPEN_3D_VIEWER = True
MAX_FRAMES = None
# -------------------------------------------------------


def run(
    input_video=INPUT_VIDEO,
    output_dir=OUTPUT_DIR,
    open_3d_viewer=None,
    progress_callback=None
):
    process.configure({
        "MAX_FEATURES": MAX_FEATURES,
        "QUALITY_LEVEL": QUALITY_LEVEL,
        "MIN_DISTANCE": MIN_DISTANCE,
        "MIN_TRACKED_POINTS": MIN_TRACKED_POINTS,
        "MIN_ESSENTIAL_INLIERS": MIN_ESSENTIAL_INLIERS,
        "MIN_POSE_VALID_POINTS": MIN_POSE_VALID_POINTS,
        "DENSE_GRID_STEP": DENSE_GRID_STEP,
        "FB_THRESHOLD": FB_THRESHOLD,
    })

    # Clear previous outputs before processing a new video.
    output_dir = Path(output_dir).resolve()
    input_path = Path(input_video).resolve()

    if input_path == output_dir or output_dir in input_path.parents:
        raise ValueError("Input video cannot be inside the output directory.")

    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    if open_3d_viewer is None:
        open_3d_viewer = OPEN_3D_VIEWER

    depth_dir = output_dir / "depth_maps"
    cloud_dir = output_dir / "point_clouds"
    depth_dir.mkdir(parents=True, exist_ok=True)
    cloud_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open input video: {input_video}")

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if PROCESS_WIDTH is None:
        width, height = original_width, original_height
    else:
        width = int(PROCESS_WIDTH)
        height = int(round(original_height * width / original_width))

    K = process.create_camera_matrix(width, height, FOCAL_MULTIPLIER)
    video_path = output_dir / "depth_analysis.mp4"
    writer = visualizer.create_video_writer(
        str(video_path), fps, PANEL_WIDTH * 2, PANEL_HEIGHT * 2 + 58
    )

    ret, first = cap.read()
    if not ret:
        cap.release()
        writer.release()
        raise RuntimeError("Could not read first frame.")

    first = cv2.resize(first, (width, height), interpolation=cv2.INTER_AREA)
    previous_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    pose_reference_gray = previous_gray.copy()
    pose_reference_index = 0
    pose_reference_points = process.detect_features(pose_reference_gray)

    optical_panel = np.zeros_like(first)
    dense_depth = np.full((height, width), np.nan, np.float32)
    sparse_depth = np.full((height, width), np.nan, np.float32)

    frame_number = 0
    pose_successes = 0
    latest_tracked = 0
    latest_dense_count = 0
    latest_status = "Waiting for first pose interval"
    latest_cloud_path = None

    print(f"Input: {input_video}")
    print(f"Original resolution: {original_width} x {original_height}")
    print(f"Processing resolution: {width} x {height}; FPS: {fps:.2f}")
    print("Approximate camera matrix:\n", K)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_number += 1
            if MAX_FRAMES is not None and frame_number >= MAX_FRAMES:
                break

            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Consecutive-frame sparse flow.
            points = process.detect_features(previous_gray)
            old, new = process.track_features(previous_gray, gray, points)
            optical_panel = process.visualize_sparse_flow(frame, old, new)

            # Pose/depth estimation at the selected frame interval.
            if frame_number - pose_reference_index >= FRAME_GAP:
                old, new = process.track_features(
                    pose_reference_gray, gray, pose_reference_points
                )
                latest_tracked = len(old)
                pose = process.estimate_camera_pose(old, new, K)
                latest_status = pose["status"]

                if pose["success"]:
                    pose_successes += 1
                    R, t = pose["R"], pose["t"]

                    sparse_depth, sparse_count, sparse_xyz, sparse_z = (
                        process.calculate_sparse_depth(
                            pose["points1"], pose["points2"], R, t, K, width, height
                        )
                    )
                    flow = process.calculate_dense_flow(pose_reference_gray, gray)
                    dense_depth, latest_dense_count, dense_xyz, dense_z = (
                        process.calculate_dense_depth(
                            pose_reference_gray, gray, flow, R, t, K, width, height
                        )
                    )

                    xyz = dense_xyz if len(dense_xyz) else sparse_xyz
                    z = dense_z if len(dense_z) else sparse_z
                    if len(xyz):
                        latest_cloud_path = cloud_dir / (
                            f"cloud_{pose_reference_index:06d}_{frame_number:06d}.ply"
                        )
                        visualizer.save_point_cloud(xyz, z, str(latest_cloud_path))

                    visualizer.save_depth_image(
                        dense_depth, str(depth_dir /
                        f"dense_depth_{pose_reference_index:06d}_{frame_number:06d}.png")
                    )
                    visualizer.save_depth_image(
                        sparse_depth, str(depth_dir /
                        f"sparse_depth_{pose_reference_index:06d}_{frame_number:06d}.png")
                    )
                else:
                    latest_dense_count = 0

                # Retain the original rolling reference-frame strategy.
                pose_reference_gray = gray.copy()
                pose_reference_index = frame_number
                pose_reference_points = process.detect_features(pose_reference_gray)

            dense_panel = visualizer.make_depth_visualization(dense_depth)
            sparse_panel = visualizer.make_depth_visualization(sparse_depth)
            status = (
                f"Frame {frame_number}/{total_frames} | tracked {latest_tracked} | "
                f"pose successes {pose_successes} | dense samples {latest_dense_count} | "
                f"{latest_status}"
            )
            combined = visualizer.create_combined_frame(
                frame, optical_panel, dense_panel, sparse_panel,
                PANEL_WIDTH, PANEL_HEIGHT, status
            )
            writer.write(combined)

            if DISPLAY:
                cv2.imshow("Classical Monocular Depth", combined)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if frame_number % 30 == 0:
                print(f"Frame {frame_number}/{total_frames} | {latest_status}")
            previous_gray = gray

    finally:
        cap.release()
        writer.release()
        if DISPLAY:
            cv2.destroyAllWindows()

    print("\nProcessing complete.")
    print("Analysis video:", video_path)
    print("Depth maps:", depth_dir)
    print("Point clouds:", cloud_dir)

    if open_3d_viewer and latest_cloud_path is not None:
        visualizer.show_point_cloud(str(latest_cloud_path))
    elif latest_cloud_path is None:
        print("No valid point cloud was generated. Try a larger FRAME_GAP or check camera motion.")

    return {
        "video": str(video_path),
        "output_dir": str(output_dir),
        "latest_point_cloud": str(latest_cloud_path) if latest_cloud_path else None,
        "pose_successes": pose_successes,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=INPUT_VIDEO)
    parser.add_argument("--output", default=OUTPUT_DIR)
    args = parser.parse_args()
    run(args.input, args.output)
