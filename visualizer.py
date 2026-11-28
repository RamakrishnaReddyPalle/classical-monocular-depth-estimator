
import os
import cv2
import numpy as np
import streamlit as st
import plotly.graph_objects as go

def show_point_cloud_streamlit(path):
    """Render an interactive 3D point cloud inside Streamlit."""

    import open3d as o3d

    cloud = o3d.io.read_point_cloud(str(path))

    if cloud.is_empty():
        st.error("The point cloud is empty.")
        return

    cloud = cloud.voxel_down_sample(voxel_size=0.005)

    points = np.asarray(cloud.points)

    if len(points) == 0:
        st.error("No valid 3D points found.")
        return

    # Limit the number of points for browser performance.
    if len(points) > 40000:
        indices = np.linspace(
            0, len(points) - 1, 40000
        ).astype(int)

        points = points[indices]

        if cloud.has_colors():
            colors = np.asarray(cloud.colors)[indices]
        else:
            colors = None

    elif cloud.has_colors():
        colors = np.asarray(cloud.colors)
    else:
        colors = None

    if colors is not None:
        colors = [
            f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
            for r, g, b in colors
        ]
    else:
        colors = "#20a4d8"

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                marker=dict(
                    size=1.5,
                    color=colors,
                    opacity=0.85
                )
            )
        ]
    )

    fig.update_layout(
        title="Monocular 3D Reconstruction",
        height=700,
        margin=dict(l=0, r=0, b=0, t=45),
        scene=dict(
            aspectmode="data",
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z"
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"scrollZoom": True}
    )

def make_depth_visualization(depth_map, output_size=None):
    """Create a high-contrast, colourful depth visualization."""
    valid = np.isfinite(depth_map) & (depth_map > 0)
    h, w = depth_map.shape

    if not np.any(valid):
        return np.zeros((h, w, 3), np.uint8)

    values = depth_map[valid]
    lo, hi = np.percentile(values, [2, 98])

    if hi <= lo:
        hi = lo + 1e-6

    normalized = np.zeros((h, w), np.float32)
    normalized[valid] = np.clip(
        (depth_map[valid] - lo) / (hi - lo), 0, 1
    )

    weight = valid.astype(np.float32)

    if output_size is None:
        output_size = (w, h)

    numerator = cv2.resize(
        normalized, output_size, interpolation=cv2.INTER_CUBIC
    )
    confidence = cv2.resize(
        weight, output_size, interpolation=cv2.INTER_CUBIC
    )

    smooth = np.clip(
        numerator / np.maximum(confidence, 1e-3), 0, 1
    )

    # Apply a mild contrast enhancement.
    smooth = np.clip((smooth - 0.5) * 1.25 + 0.5, 0, 1)

    color = cv2.applyColorMap(
        (smooth * 255).astype(np.uint8),
        cv2.COLORMAP_TURBO
    )

    color[confidence < 0.025] = (12, 12, 18)

    return color


def add_title(image, title, height=42):
    out = image.copy()

    cv2.rectangle(
        out, (0, 0), (out.shape[1], height),
        (22, 25, 35), -1
    )

    cv2.putText(
        out, title, (16, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (255, 255, 255), 2, cv2.LINE_AA
    )

    return out


def resize_panel(image, width, height):
    return cv2.resize(
        image, (width, height),
        interpolation=cv2.INTER_AREA
    )


def create_combined_frame(original, optical_flow, dense_depth, sparse_depth,
                          panel_width, panel_height, status):

    # Allocate a large depth panel and two smaller side panels.
    canvas_width = panel_width * 2
    canvas_height = panel_height * 2

    side_width = int(canvas_width * 0.30)
    depth_width = canvas_width - side_width

    depth_height = canvas_height
    small_height = canvas_height // 2

    # Main depth visualization.
    depth = resize_panel(
        dense_depth, depth_width, depth_height
    )
    depth = add_title(depth, "DENSE RELATIVE DEPTH")

    # Original frame and optical flow on the right.
    original_small = resize_panel(
        original, side_width, small_height
    )
    original_small = add_title(
        original_small, "ORIGINAL FRAME"
    )

    flow_small = resize_panel(
        optical_flow, side_width, small_height
    )
    flow_small = add_title(
        flow_small, "SPARSE OPTICAL FLOW"
    )

    # Combine the three panels.
    side_panel = np.vstack((original_small, flow_small))
    canvas = np.hstack((depth, side_panel))

    # Add status information at the bottom.
    bar = np.zeros((58, canvas_width, 3), np.uint8)

    cv2.putText(
        bar, status[:180], (18, 37),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62, (235, 240, 248), 1, cv2.LINE_AA
    )

    return np.vstack((canvas, bar))


def create_video_writer(path, fps, width, height):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    for codec in ("mp4v", "avc1"):
        writer = cv2.VideoWriter(
            path,
            cv2.VideoWriter_fourcc(*codec),
            fps,
            (width, height)
        )

        if writer.isOpened():
            print(f"Video writer initialized: {codec}")
            return writer

        writer.release()

    raise RuntimeError(
        "Could not initialize MP4 writer. Check OpenCV/FFmpeg."
    )


def save_depth_image(depth_map, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cv2.imwrite(path, make_depth_visualization(depth_map))


def save_point_cloud(points3d, depths, path):
    """Write a coloured PLY point cloud."""
    if len(points3d) == 0:
        return False

    os.makedirs(os.path.dirname(path), exist_ok=True)

    lo, hi = np.percentile(depths, [2, 98])
    normalized = np.clip(
        (depths - lo) / max(hi - lo, 1e-6), 0, 1
    )

    colors = cv2.applyColorMap(
        (normalized * 255).astype(np.uint8).reshape(-1, 1),
        cv2.COLORMAP_TURBO
    ).reshape(-1, 3)[:, ::-1]

    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points3d)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for p, c in zip(points3d, colors):
            f.write(
                f"{p[0]} {p[1]} {p[2]} "
                f"{int(c[0])} {int(c[1])} {int(c[2])}\n"
            )

    return True


def show_point_cloud(path):
    try:
        import open3d as o3d
    except ImportError:
        print("Install Open3D with: pip install open3d")
        print("You can also open the PLY file in MeshLab or CloudCompare.")
        return

    cloud = o3d.io.read_point_cloud(path)

    if cloud.is_empty():
        print("Point cloud is empty:", path)
        return

    cloud = cloud.voxel_down_sample(voxel_size=0.005)
    cloud.estimate_normals()

    o3d.visualization.draw_geometries(
        [cloud],
        window_name="Monocular 3D Reconstruction",
        width=1440,
        height=900
    )
