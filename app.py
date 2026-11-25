
import subprocess
import tempfile
from pathlib import Path

import cv2
import streamlit as st
import imageio_ffmpeg

import main
import visualizer


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "data" / "output"

GITHUB_URL = (
    "https://github.com/RamakrishnaReddyPalle/"
    "classical-monocular-depth-estimator"
)

MAX_UPLOAD_MB = 500


st.set_page_config(
    page_title="Classical Monocular Depth Estimator",
    page_icon="🎥",
    layout="wide"
)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

if "result" not in st.session_state:
    st.session_state.result = None

if "browser_video" not in st.session_state:
    st.session_state.browser_video = None

if "input_video_bytes" not in st.session_state:
    st.session_state.input_video_bytes = None

if "input_video_name" not in st.session_state:
    st.session_state.input_video_name = None


# --------------------------------------------------
# VIDEO CONVERSION
# --------------------------------------------------

def convert_video_for_browser(input_path):
    input_path = Path(input_path)
    output_path = input_path.parent / "depth_analysis_browser.mp4"

    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-i", str(input_path),
        "-an",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output_path)
    ]

    subprocess.run(
        command,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE
    )

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError("Browser-compatible video was not generated.")

    return output_path


# --------------------------------------------------
# VALIDATION AND RESET
# --------------------------------------------------

def validate_video(upload):
    if upload is None:
        return False, "Please upload a video."

    if Path(upload.name).suffix.lower() not in {
        ".mp4", ".mov", ".avi", ".mkv"
    }:
        return False, "Unsupported video format."

    if upload.size == 0:
        return False, "The uploaded video is empty."

    if upload.size > MAX_UPLOAD_MB * 1024 * 1024:
        return False, f"Maximum upload size is {MAX_UPLOAD_MB} MB."

    return True, "Valid video."


def reset_application():
    st.session_state.result = None
    st.session_state.browser_video = None
    st.session_state.input_video_bytes = None
    st.session_state.input_video_name = None
    st.session_state.uploader_key += 1
    st.rerun()


# --------------------------------------------------
# HEADER
# --------------------------------------------------

header_left, header_right = st.columns([2.2, 1])

with header_left:
    st.title("Classical Monocular Depth Estimator")

    st.caption(
        "Classical Computer Vision | Monocular Depth Estimation "
        "| 3D Reconstruction"
    )

    st.markdown(
        f"""
        <a href="{GITHUB_URL}" target="_blank" style="
            display:inline-flex;
            align-items:center;
            gap:9px;
            padding:9px 22px;
            margin-top:5px;
            background:#238636;
            color:white;
            border-radius:7px;
            text-decoration:none;
            font-weight:600;
        ">
            <img src="https://cdn.simpleicons.org/github/ffffff"
                 width="20" height="20">
            &lt;&gt; Code
        </a>
        """,
        unsafe_allow_html=True
    )

with header_right:
    st.markdown(
        """
        <div style="text-align:right; padding-top:10px;">
            <div style="font-size:23px; font-weight:900;">
                VIDEO ANALYTICS
            </div>
            <div style="font-size:17px; color:#9ca3af;">
                Course Project
            </div>
            <div style="font-size:14px; margin-top:8px; line-height:1.9;">
                Ramakrishna Reddy Palle<br>
                Ashmit Kaur<br>
                Niloy
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.divider()


# --------------------------------------------------
# MAIN DASHBOARD
# --------------------------------------------------

left_col, right_col = st.columns(
    [0.85, 1.35],
    gap="large"
)


# --------------------------------------------------
# LEFT: INPUT AND PROCESSING
# --------------------------------------------------

with left_col:

    st.subheader("Input Video")

    uploaded_video = st.file_uploader(
        "Choose a video",
        type=["mp4", "mov", "avi", "mkv"],
        key=f"video_uploader_{st.session_state.uploader_key}"
    )

    if uploaded_video is not None:

        valid, message = validate_video(uploaded_video)

        if not valid:
            st.error(message)

        else:
            st.caption(
                f"{uploaded_video.name} | "
                f"{uploaded_video.size / (1024 * 1024):.2f} MB"
            )

            st.video(uploaded_video)

            if st.button(
                "Run Depth Estimation",
                type="primary",
                use_container_width=True
            ):

                st.session_state.result = None
                st.session_state.browser_video = None

                temp_path = None

                status_box = st.status(
                    "Initializing reconstruction...",
                    expanded=True
                )

                try:
                    suffix = Path(uploaded_video.name).suffix.lower()

                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=suffix
                    ) as temp_file:
                        temp_file.write(uploaded_video.getbuffer())
                        temp_path = Path(temp_file.name)

                    cap = cv2.VideoCapture(str(temp_path))

                    if not cap.isOpened():
                        raise ValueError(
                            "Cannot open the uploaded video. "
                            "It may be corrupted or unsupported."
                        )

                    total_frames = int(
                        cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    )
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)

                    cap.release()

                    if total_frames <= 0 or width <= 0 or height <= 0:
                        raise ValueError("Invalid video metadata.")

                    st.session_state.input_video_bytes = (
                        uploaded_video.getvalue()
                    )
                    st.session_state.input_video_name = uploaded_video.name

                    status_box.write(
                        f"Input: {width} × {height} | "
                        f"{fps:.2f} FPS | {total_frames} frames"
                    )

                    # Receive live diagnostic updates from main.py.
                    def update_progress(progress, message):
                        status_box.update(
                            label=message,
                            state="running",
                            expanded=True
                        )
                        status_box.write(
                            f"Processing progress: {progress * 100:.1f}%"
                        )

                    with st.spinner(
                        "Running camera estimation and depth reconstruction..."
                    ):
                        result = main.run(
                            input_video=str(temp_path),
                            output_dir=str(OUTPUT_DIR),
                            open_3d_viewer=False,
                            progress_callback=update_progress
                        )

                    status_box.update(
                        label="Reconstruction completed. Preparing video...",
                        state="running"
                    )

                    if not Path(result["video"]).exists():
                        raise FileNotFoundError(
                            "The processed video was not generated."
                        )

                    browser_video = convert_video_for_browser(
                        result["video"]
                    )

                    st.session_state.result = result
                    st.session_state.browser_video = str(browser_video)

                    status_box.update(
                        label="Processing completed successfully.",
                        state="complete",
                        expanded=False
                    )

                    st.success("Depth estimation completed.")

                except subprocess.CalledProcessError as e:
                    status_box.update(
                        label="Video conversion failed.",
                        state="error"
                    )
                    st.error(
                        e.stderr.decode(errors="replace")[-2500:]
                        if e.stderr else str(e)
                    )

                except Exception as e:
                    status_box.update(
                        label="Reconstruction failed.",
                        state="error"
                    )
                    st.error(str(e))
                    st.exception(e)

                finally:
                    if temp_path is not None and temp_path.exists():
                        temp_path.unlink()

    if st.session_state.result is not None:
        st.divider()

        if st.button(
            "Upload Another Video",
            use_container_width=True
        ):
            reset_application()


# --------------------------------------------------
# RIGHT: OUTPUT ARTIFACTS
# --------------------------------------------------

with right_col:

    st.subheader("Processed Depth Video")

    if st.session_state.result is not None:

        browser_path = Path(st.session_state.browser_video)

        if browser_path.exists():
            st.video(browser_path.read_bytes(), format="video/mp4")
        else:
            st.error("Processed video is unavailable.")

    else:
        st.info("The processed depth video will appear here.")

    st.divider()

    st.subheader("3D Reconstruction")

    if st.session_state.result is not None:

        cloud_path = st.session_state.result.get("latest_point_cloud")

        if cloud_path and Path(cloud_path).exists():

            st.caption(
                "Inspect the reconstructed point cloud using the "
                "native Open3D viewer."
            )

            if st.button(
                "Open Interactive 3D Viewer",
                type="primary",
                use_container_width=True
            ):
                try:
                    visualizer.show_point_cloud(cloud_path)
                except Exception as e:
                    st.error(f"Could not open the 3D viewer: {e}")

        else:
            st.warning(
                "No valid point cloud was generated. "
                "Try a recording with more camera translation."
            )

    else:
        st.info("The 3D point cloud will appear here after processing.")


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Classical Monocular Depth Estimator | Video Analytics Course Project"
)
