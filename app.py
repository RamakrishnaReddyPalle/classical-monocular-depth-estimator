import subprocess
import tempfile
from pathlib import Path

import cv2
import streamlit as st
import imageio_ffmpeg
from PIL import Image

import main
import visualizer

import base64
import mimetypes


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "data" / "output"
ASSETS_DIR = BASE_DIR / "assets"

GITHUB_URL = (
    "https://github.com/RamakrishnaReddyPalle/"
    "classical-monocular-depth-estimator"
)

# Replace this placeholder with the direct GitHub PPT release-asset URL.
PPT_URL = "https://github.com/RamakrishnaReddyPalle/classical-monocular-depth-estimator/assets/123456/Classical_Monocular_Depth_Estimator.pptx"

MAX_UPLOAD_MB = 500

st.set_page_config(
    page_title="Classical Monocular Depth Estimator",
    page_icon="🎥",
    layout="wide"
)


# --------------------------------------------------
# SESSION STATE AND PAGE ROUTING
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

# Query-string navigation gives the report its own URL.
# Browser Back returns to the app page.
CURRENT_PAGE = st.query_params.get("page", "app")


# --------------------------------------------------
# SHARED STYLING
# --------------------------------------------------

st.markdown(
    """
    <style>
    .top-link {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        padding: 9px 16px;
        margin: 5px 7px 4px 0;
        border-radius: 7px;
        color: #ffffff !important;
        text-decoration: none !important;
        font-weight: 650;
        font-size: 14px;
        white-space: nowrap;
    }
    .code-link { background: #238636; }
    .report-link { background: #d97706; }
    .ppt-link { background: #475569; }
    .report-back {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: #d97706;
        color: white !important;
        padding: 9px 16px;
        border-radius: 7px;
        text-decoration: none !important;
        font-weight: 650;
        margin: 4px 0 18px 0;
    }
    .report-hero {
        padding: 28px 30px;
        border-radius: 14px;
        background: linear-gradient(120deg, #111827, #1e3a5f);
        border: 1px solid #334155;
        margin: 12px 0 24px 0;
    }
    .report-kicker {
        color: #fbbf24;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    .report-hero h1 {
        color: #f8fafc;
        margin: 10px 0 8px 0;
        font-size: 34px;
    }
    .report-hero p { color: #cbd5e1; margin-bottom: 0; }
    .report-caption {
        color: #94a3b8;
        font-size: 13px;
    }
    .block-container {
        max-width: 1600px;
        padding-left: 2.5rem;
        padding-right: 2.5rem;
    }
     .report-body {
        font-size: 19px;
        line-height: 1.78;
    }
    .stMarkdown p, .stMarkdown li {
        font-size: 19px;
        line-height: 1.78;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        line-height: 1.35;
    }
    /* Keep the report index fixed while the report scrolls */
    .st-key-report_index {
        position: fixed !important;
        top: 110px !important;

        /* Align with the report's left column */
        left: max(
            2.5rem,
            calc((100vw - 1600px) / 2 + 2.5rem)
        ) !important;

        width: calc(
            (min(100vw, 1600px) - 5rem) * 0.20
        ) !important;

        z-index: 1000;
        align-self: flex-start;
    }

    .report-index {
        padding: 12px 10px;
        border-right: 1px solid rgba(128,128,128,.35);
        background: var(--background-color);
    }
    @media (max-width: 768px) {
        .st-key-report_index {
            position: static !important;
            width: 100% !important;
        }
    }
    .report-index-title {
        font-size: 18px;
        font-weight: 750;
        margin-bottom: 12px;
    }
    .report-index a {
        display: block;
        padding: 7px 8px;
        margin: 2px 0;
        border-radius: 6px;
        color: inherit !important;
        text-decoration: none !important;
        font-size: 15px;
        line-height: 1.45;
    }
    .report-index a:hover {
        background: rgba(217,119,6,.13);
        color: #d97706 !important;
    }
    .equation-label {
        color: #d97706;
        font-weight: 700;
        font-size: 13px;
        margin-bottom: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# SHARED HEADER LINKS
# --------------------------------------------------

def render_header_links():
    st.markdown(
        f"""
        <a class="top-link code-link" href="{GITHUB_URL}" target="_blank">
            <span>⌘</span> Code
        </a>
        <a class="top-link report-link" href="?page=report">
            <span>▤</span> Report
        </a>
        <a class="top-link ppt-link" href="{PPT_URL}" target="_blank">
            <span>▣</span> PPT
        </a>
        """,
        unsafe_allow_html=True
    )


def render_project_header():
    header_left, header_right = st.columns([2.2, 1])

    with header_left:
        st.title("Classical Monocular Depth Estimator")
        st.caption(
            "Classical Computer Vision | Monocular Depth Estimation "
            "| 3D Reconstruction"
        )
        render_header_links()

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
# REPORT HELPERS
# --------------------------------------------------


def report_image(filename, caption, width=900):
    """Display the original PNG or SVG directly in the browser."""

    image_path = ASSETS_DIR / filename

    if not image_path.exists():
        st.caption(f"Add `assets/{filename}` — {caption}")
        return

    extension = image_path.suffix.lower()

    if extension == ".svg":
        mime_type = "image/svg+xml"
    elif extension == ".png":
        mime_type = "image/png"
    else:
        mime_type = mimetypes.guess_type(image_path.name)[0]

    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(
            image_file.read()
        ).decode("utf-8")

    image_html = f"""
    <div style="width:100%; text-align:center; margin: 12px 0 6px 0;">
        <img
            src="data:{mime_type};base64,{encoded_image}"
            style="
                display:block;
                width:{width}px;
                max-width:100%;
                height:auto;
                object-fit:contain;
                image-rendering:auto;
                margin:0 auto;
            "
        >
        <div style="
            font-size:14px;
            color:var(--text-color);
            opacity:0.75;
            margin-top:8px;
            line-height:1.5;
        ">
            {caption}
        </div>
    </div>
    """

    st.markdown(image_html, unsafe_allow_html=True)


def report_section(number, heading):
    st.markdown(
        f'<div id="report-section-{number}"></div>',
        unsafe_allow_html=True
    )
    st.markdown(f"## {number}. {heading}")


def render_report():
    index_col, report_col = st.columns([0.22, 0.78], gap="large")

    with index_col:
        with st.container(key="report_index"):
            st.markdown(
                """
                <div class="report-index">
                    <div class="report-index-title">Report Contents</div>
                    <a href="#report-section-1">1. Introduction</a>
                    <a href="#report-section-2">2. Literature Review</a>
                    <a href="#report-section-3">3. Methodology</a>
                    <a href="#report-section-4">4. Experimental Setup</a>
                    <a href="#report-section-5">5. Results & Discussion</a>
                    <a href="#report-section-6">6. Conclusion</a>
                    <a href="#report-section-7">7. References</a>
                    <a href="#report-section-8">8. Presentation & Demo</a>
                </div>
                """,
                unsafe_allow_html=True
            )

    with report_col:
        st.markdown(
            '<a class="report-back" href="?page=app">← Back to Depth Estimator</a>',
            unsafe_allow_html=True
        )
        st.markdown(
            """
            <div class="report-hero">
                <div class="report-kicker">Video Analytics · Course Project</div>
                <h1>Classical Monocular Depth Estimation</h1>
                <p>
                    Sparse geometric reconstruction and dense optical-flow-assisted
                    depth visualisation from monocular video.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            **Team**

            Ramakrishna Reddy Palle  ·  Ashmit Kaur  ·  Niloy Dev

            **Project repository:** [GitHub](https://github.com/RamakrishnaReddyPalle/classical-monocular-depth-estimator)
            """
        )

        report_section("1", "Introduction")
        st.markdown(
            """
            Recovering scene structure from a moving monocular camera is a
            fundamental computer-vision problem. A single camera is inexpensive
            and widely available, but it does not directly provide metric depth.
            This project studies a classical geometry-based pipeline that estimates
            relative 3D structure from consecutive video frames.

            **Motivation and applications.** Monocular reconstruction is relevant
            to camera-motion analysis, scene understanding, visual navigation,
            augmented reality and low-cost 3D inspection.

            **Problem statement.** Given a monocular video, estimate feature
            correspondences between frames, recover relative camera motion, and
            triangulate scene points to obtain sparse depth samples. A dense
            optical-flow path supplies additional correspondences for a denser
            relative-depth visualisation.

            **Contributions**
            - A classical feature-tracking and two-view geometry pipeline.
            - Essential-matrix estimation with RANSAC and relative pose recovery.
            - Sparse triangulation and a forward-backward-checked dense-flow path.
            - Depth-video and coloured point-cloud outputs with an interactive
              Open3D inspection option.
            """
        )

        report_section("2", "Brief Literature Review")
        st.markdown(
            """
            The implementation follows established classical methods rather than
            a learned depth network. Shi–Tomasi feature selection provides
            trackable image corners; pyramidal Lucas–Kanade estimates their
            displacement across frames. Two-view geometry uses the essential
            matrix to constrain corresponding image points and recover relative
            camera pose. Triangulation then estimates 3D points up to the
            monocular scale ambiguity. Farnebäck optical flow provides a dense
            displacement field, while forward-backward consistency rejects
            unreliable samples.

            The methods and their original references are listed in Section 7.
            """
        )

        report_section("3", "Methodology")

        st.markdown("### 3.1 Camera model and image preparation")
        st.markdown(
            """
            Each frame is represented using a pinhole camera model. The intrinsic
            matrix is approximated because a calibrated camera matrix is not
            assumed by this project.
            """
        )
        st.latex(
            r"K = \begin{bmatrix}"
            r"f_x & 0 & c_x \\"
            r"0 & f_y & c_y \\"
            r"0 & 0 & 1"
            r"\end{bmatrix}"
        )
        st.markdown(
            """
            Here, \(f_x,f_y\) are focal lengths in pixel units and
            \((c_x,c_y)\) is the principal point. Approximate intrinsics affect
            the accuracy of pose and depth estimates; calibration is a natural
            improvement.
            """
        )
        report_image("input_video_frames.png",
                     "Input-video samples at different points in the sequence.",
                     width=900)
        st.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True)
        report_image("undistortion_comparison.png",
                     "Original frame and the corresponding undistorted image.",
                     width=900)


        st.markdown("<div style='height: 60px;'></div>", unsafe_allow_html=True)
        st.markdown("### 3.2 Shi–Tomasi feature detection")
        st.markdown(
            """
            Candidate points are selected using the minimum eigenvalue of the
            local gradient matrix. For a window \(W\), define
            """
        )
        st.latex(
            r"M = \sum_{(x,y)\in W}"
            r"\begin{bmatrix}"
            r"I_x^2 & I_x I_y \\"
            r"I_x I_y & I_y^2"
            r"\end{bmatrix},"
            r"\qquad R = \min(\lambda_1,\lambda_2)"
        )
        st.markdown(
            """
            Points with a sufficiently large \(R\) are retained, subject to
            feature-spacing and maximum-feature constraints.
            """
        )
        st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
        report_image("shi_tomasi_features.png",
                     "Shi–Tomasi corners detected on the undistorted first frame.",
                     width=900)


        st.markdown("<div style='height: 60px;'></div>", unsafe_allow_html=True)
        st.markdown("### 3.3 Pyramidal Lucas–Kanade tracking")
        st.markdown(
            """
            The pyramidal Lucas–Kanade method tracks a point from frame \(t\)
            to frame \(t+1\) by finding the displacement that minimises the
            local brightness-constancy error:
            """
        )
        st.latex(
            r"E(u,v)=\sum_{(x,y)\in W}"
            r"\left[I_{t+1}(x+u,y+v)-I_t(x,y)\right]^2"
        )
        st.markdown(
            """
            The pyramid allows larger apparent motion to be estimated at coarser
            resolutions and refined at finer resolutions. Tracks that fail the
            tracker validity checks are discarded.
            """
        )
        st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
        report_image("klt_tracking.svg",
                     "Illustration of matched feature points across consecutive frames.",
                     width=500)
        st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
        report_image("klt_feature_matches.png",
                     "Tracked feature correspondences between two consecutive frames.",
                     width=900)



        st.markdown("<div style='height: 60px;'></div>", unsafe_allow_html=True)
        # Section 3.4: keep the heading separate; prose left and two compact
        # illustrations stacked on the right.
        st.markdown("### 3.4 Essential matrix and relative pose")
        pose_text, pose_images = st.columns([1, 1], gap="medium")
        with pose_text:
            st.markdown(
                """
                For corresponding homogeneous image points \(x_1\) and \(x_2\),
                the epipolar constraint is:
                """
            )
            st.latex(r"x_2^T E x_1 = 0")
            st.markdown(
                """
                The essential matrix is related to relative rotation \(R\) and
                translation direction \(t\) by:
                """
            )
            st.latex(r"E=[t]_\times R")
            st.markdown(
                """
                RANSAC estimates a robust essential matrix from matched points.
                Decomposition of \(E\), followed by cheirality validation, selects
                a physically consistent relative pose. Translation magnitude remains
                unknown in monocular reconstruction.
                """
            )
        with pose_images:
            report_image(
                "pose_validation.svg",
                "Pose-validation decision path using tracked points, essential-matrix inliers and pose-valid points.",
                width=520
            )
        st.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True)
        report_image(
            "camera_trajectory.png",
            "Estimated relative camera trajectory. The trajectory is expressed in arbitrary monocular scale.",
            width=520
        )

        st.markdown("<div style='height: 60px;'></div>", unsafe_allow_html=True)
        # Section 3.5: text left, triangulation figure right.
        st.markdown("### 3.5 Sparse triangulation")
        triangulation_text, triangulation_figure = st.columns([1, 1], gap="medium")
        with triangulation_text:
            st.markdown(
                """
                With projection matrices
                """
            )

            st.latex(r"P_1 = K[I \mid 0]")

            st.markdown("and")

            st.latex(r"P_2 = K[R \mid t]")

            st.markdown(
                """
                matched image points are triangulated:
                """
            )
            st.latex(
                r"\tilde{X}=\operatorname{triangulate}(P_1,P_2,x_1,x_2),"
                r"\qquad Z = X_3"
            )
            st.markdown(
                """
                The resulting 3D points provide sparse relative-depth samples.
                Their absolute scale cannot be recovered from a monocular sequence
                without an additional scale reference.
                """
            )
        with triangulation_figure:
            report_image(
                "triangulation_geometry.svg",
                "Two-view triangulation geometry and the unknown monocular baseline scale.",
                width=400
            )


        st.markdown("<div style='height: 60px;'></div>", unsafe_allow_html=True)
        # Section 3.6: dense-flow pipeline figure first on the right;
        # the two existing result figures remain full report-width below.
        st.markdown("### 3.6 Dense optical flow and consistency filtering")
        flow_text, flow_pipeline_figure = st.columns([1, 1], gap="medium")
        with flow_text:
            st.markdown(
                """
                Farnebäck flow estimates a displacement vector at each pixel. A
                regular image grid is sampled, and forward-backward consistency is
                used to reject unreliable matches. For a point \(p_t\), the check is:
                """
            )
            st.latex(
                r"e_{FB}=\left\|p_t-\hat{p}_t\right\|_2"
                r"<\tau_{FB},\qquad \tau_{FB}=1.5\;\text{pixels}"
            )
            st.markdown(
                """
                Surviving correspondences are triangulated using the relative pose
                estimated for the frame pair. The resulting samples support the
                denser depth visualisation; interpolation used for display does not
                create additional measured depth.
                """
            )
        with flow_pipeline_figure:
            report_image(
                "dense_flow_pipeline.svg",
                "Dense path: Farnebäck flow, grid sampling, forward-backward filtering and triangulation.",
                width=280
            )
        st.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True)
        report_image(
            "dense_optical_flow_first_frame.png",
            "Example first-frame dense optical-flow visualisation.",
            width=1000
        )
        st.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True)
        report_image(
            "dense_optical_flow_samples.png",
            "Dense-flow visualisations at selected frame pairs.",
            width=1000
        )


        st.markdown("<div style='height: 70px;'></div>", unsafe_allow_html=True)
        st.markdown("### 3.7 Overall pipeline")
        st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
        report_image(
            "pipeline_overview.svg",
            "Conceptual overview from video and camera motion to triangulated depth.",
            width=900
        )
        st.markdown("<div style='height: 60px;'></div>", unsafe_allow_html=True)
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        report_image(
            "pipeline_architecture.svg",
            "Implemented sparse and dense processing paths.",
            width=900
        )

        st.markdown("<div style='height: 60px;'></div>", unsafe_allow_html=True)
        report_section("4", "Experimental Setup")
        st.markdown(
            """
            ***Input data:*** 
            - The application accepts user-supplied monocular video
            files in MP4, MOV, AVI and MKV formats. The current project does not
            bundle a labelled benchmark dataset; results are therefore presented
            as qualitative reconstructions rather than benchmark accuracy scores.

            ***Implementation platform:*** 
            - Python, OpenCV, NumPy, Streamlit,
            imageio-ffmpeg and Open3D. OpenCV performs image processing and
            geometric estimation; FFmpeg prepares browser-compatible output video;
            Open3D is used for interactive point-cloud inspection.

            ***Baseline and comparison:*** 
            - The sparse KLT + essential-matrix +
            triangulation path is the geometric reconstruction baseline. The
            Farnebäck path provides denser image correspondences. No learned
            monocular-depth model or external benchmark baseline is claimed in
            this implementation.
            """
        )
        st.markdown("<div style='height: 60px;'></div>", unsafe_allow_html=True)
        st.markdown("### Project structure")
        st.code(
            """classical-monocular-depth-estimator/
├── app.py
├── main.py
├── process.py
├── visualizer.py
├── assets/
└── data/
    ├── input/
    └── output/""",
            language="text"
        )
        st.markdown("<div style='height: 60px;'></div>", unsafe_allow_html=True)
        st.markdown("### Principal configuration")
        config = {
            "Parameter": [
                "Frame gap",
                "Maximum Shi–Tomasi features",
                "Feature quality level",
                "Minimum feature distance",
                "Minimum tracked points",
                "Minimum essential-matrix inliers",
                "Minimum pose-valid points",
                "Dense-flow grid step",
                "Forward-backward threshold",
                "Focal multiplier"
            ],
            "Value": [
                "8 frames", "2000", "0.005", "5 px", "30",
                "20", "15", "3 px", "1.5 px", "1.2"
            ]
        }
        st.table(config)

        st.markdown("<div style='height: 60px;'></div>", unsafe_allow_html=True)
        report_section("5", "Results and Discussion")
        st.markdown(
            """
            **Qualitative observations**
            - Repeated architectural edges and textured surfaces provide useful
              feature locations for tracking.
            - The camera trajectory illustrates accumulated relative motion,
              but is not a metric trajectory.
            - Sparse triangulation produces depth only where reliable
              correspondences and sufficient parallax are available.
            - Dense flow increases correspondence coverage, while the
              forward-backward test removes inconsistent samples.
            - The depth video is a visualisation of relative depth, not a
              calibrated metric depth map.
            """
        )

        st.markdown(
            """
            **Failure cases and limitations**
            - **Low parallax or rotation-only motion:** depth becomes poorly
              constrained when the camera has little translation.
            - **Textureless regions:** few stable corners are available to track.
            - **Motion blur and rapid motion:** feature tracking and optical flow
              can become inconsistent.
            - **Dynamic objects and occlusion:** scene motion may violate the
              static-scene assumption used by two-view geometry.
            - **Approximate intrinsics:** focal-length error affects pose and
              triangulated depth.
            - **Scale ambiguity:** monocular translation and depth are recovered
              only up to an unknown global scale.
            """
        )

        report_section("6", "Conclusion and Future Work")
        st.markdown(
            """
            - This project implements a classical monocular reconstruction pipeline
            combining Shi–Tomasi feature detection, pyramidal Lucas–Kanade
            tracking, essential-matrix pose recovery, triangulation and
            Farnebäck dense optical flow. It produces a relative-depth video
            and coloured point clouds for visual inspection.

            **Future work**
            - Future improvements could include calibrating the camera intrinsics 
            instead of relying on approximations, improving outlier rejection under
            motion blur, occlusion, and dynamic scenes, and estimating and aligning
            a global camera trajectory across frame intervals. 
            The system could also be extended with quantitative evaluation using sequences with 
            ground-truth depth or camera poses, along with scale recovery when known scene measurements 
            are available.
            """
        )


        st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
        report_section("7", "References")
        references = [
            "Shi, J. and Tomasi, C. (1994). Good Features to Track. IEEE Conference on Computer Vision and Pattern Recognition (CVPR).",
            "Lucas, B. D. and Kanade, T. (1981). An Iterative Image Registration Technique with an Application to Stereo Vision. IJCAI.",
            "Farnebäck, G. (2003). Two-Frame Motion Estimation Based on Polynomial Expansion. Scandinavian Conference on Image Analysis.",
            "Longuet-Higgins, H. C. (1981). A Computer Algorithm for Reconstructing a Scene from Two Projections. Nature, 293, 133–135.",
            "Hartley, R. (1997). In Defense of the Eight-Point Algorithm. IEEE Transactions on Pattern Analysis and Machine Intelligence, 19(6), 580–593.",
            "Hartley, R. and Zisserman, A. (2004). Multiple View Geometry in Computer Vision (2nd ed.). Cambridge University Press.",
            "OpenCV Documentation. Camera Calibration and 3D Reconstruction; Optical Flow; Feature Detection and Description. https://docs.opencv.org/"
        ]
        for i, reference in enumerate(references, start=1):
            st.markdown(f"{i}. {reference}")

        st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
        report_section("8", "Presentation and Demonstration")
        st.markdown(
            f"""
            - **Presentation (PPT):** [Open presentation]({PPT_URL})
            - **Video demonstration:** [Watch the project demo](https://github.com/user-attachments/assets/09ff2702-0b46-4ce7-a9a7-445fb25fea12)
            """
        )

        st.divider()


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
# REPORT PAGE
# --------------------------------------------------

if CURRENT_PAGE == "report":
    render_report()

else:
    # --------------------------------------------------
    # MAIN APPLICATION PAGE
    # --------------------------------------------------

    render_project_header()

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
