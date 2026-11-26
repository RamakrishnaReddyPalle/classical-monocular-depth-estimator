# **Classical Monocular Depth Estimator**
- Course Project - Video Analytics
- Team: Ramakrishna Reddy, Ashmit Kaur, Niloy

Classical CV pipeline using Shi-Tomasi features, pyramidal Lucas-Kanade tracking, essential-matrix pose recovery, Farnebäck dense optical flow and triangulation.

### **Demo**

<p align="center">
  <video src="https://github.com/user-attachments/assets/09ff2702-0b46-4ce7-a9a7-445fb25fea12" controls muted playsinline width="100%">
    Your browser does not support embedded videos.
  </video>
</p>

### **Structure**

```text
Root/
├── process.py
├── visualizer.py
├── main.py
└── data/
    ├── input/
    └── output/
```

Put the input video at `data/input/<input>.mp4`.

### **Install**

```bash
python -m venv .venv
# Windows PowerShell
.venv\\Scripts\\Activate.ps1
# Linux
source .venv/bin/activate

pip install numpy opencv-python
# Optional interactive 3D viewer
pip install open3d
```

### **Run**

```bash
python main.py
```

Or provide paths:

```bash
python main.py --input data/input/input.mp4 --output data/output
```

Outputs include `depth_analysis.mp4`, depth-map PNGs, and coloured PLY point clouds. Open3D opens the latest successful point cloud interactively; PLY files can also be opened in MeshLab or CloudCompare.

### **Important limitations**

- The focal length is approximated. Calibrated intrinsics improve accuracy.
- Monocular translation scale is unknown; depth is relative, not metric.
- Each PLY is one pose-reference interval. This version does not estimate a global camera trajectory, so it does not merge intervals into a single consistent scene.
- The smoother heatmap interpolates between sparse triangulated samples for display. Interpolated pixels are not additional measured depth.
