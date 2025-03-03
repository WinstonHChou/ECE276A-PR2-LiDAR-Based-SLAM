# ECE276A-PR2-LiDAR-Based-SLAM
WI 25 ECE 276A Project 2: LiDAR Based SLAM

## Course Overview
This is Project 2 for [ECE 276A: Sensing & Estimation in Robotics](https://natanaso.github.io/ece276a/) at UCSD, taught by Professor [Nikolay Atanasov](https://natanaso.github.io/).

## Project Description
The project involves processing IMU, ecode, and RGBD camera data to: .......................

## Prerequisites 
The code is only tested with miniconda environment in **WSL2** to ensure that `open3d` can draw 3D pictures. If you're using Linux machine or others, **Please make sure your `open3d` works as it should.**
- Install [WSL2](https://dev.to/brayandiazc/install-wsl-from-the-microsoft-store-111h) and [External X Server (if needed)](https://www.google.com/search?q=VcXsrv)
- Miniconda Installed: https://docs.anaconda.com/miniconda/install/#quick-command-line-install
    ```bash
    mkdir -p ~/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm ~/miniconda3/miniconda.sh
    ```
    After installing, close and reopen your terminal application or refresh it by running the following command:
    ```bash
    source ~/miniconda3/bin/activate
    conda init --all
    ```
- Use `conda` to create a `python3.10` virtual environment (`ece276a_pr2`), and install required packages:
    ```bash
    conda create -n ece276a_pr2 python=3.10
    conda install -c conda-forge libstdcxx-ng -n ece276a_pr2
    conda activate ece276a_pr2
    pip3 install -r requirements.txt
    ```
- Whenever creating a **new terminal session**, do:
    ```bash
    conda deactivate
    conda activate ece276a_pr2
    export XDG_SESSION_TYPE=x11
    ```
- If you **cannot open any 3D model through `open3d`**, and any error happens as follows, 
    ```python
    [Open3D WARNING] GLFW Error: Wayland: The platform does not support setting the window position
    [Open3D WARNING] Failed to initialize GLEW.
    [Open3D WARNING] [DrawGeometries] Failed creating OpenGL window.
    ```
    **Check the following GitHub Issues: [Open3D Github Issue #6872](https://github.com/isl-org/Open3D/issues/6872), [Open3D Github Issue #5126](https://github.com/isl-org/Open3D/issues/5126)**

## Dataset Setup
1. Download the **Kinect dataset** from [OneDrive](https://ucsdcloud-my.sharepoint.com/:u:/g/personal/w3chou_ucsd_edu/ERWl0hGlKzVGl9aSChewQgEB0XdA7LNLz2cW2ncBS63aZw?e=1Dy1Ai)
2. Organize the folder structure as follows:
    ```text
    .
    └── data
      ├── <all npz files>
      ├── dataRGBD
      │ ├── Disparity20
      │ ├── Disparity21
      │ ├── RGB20
      │ └── RGB21
      └── README.md
    ```

## Running the Project

### ICP (Iterative Closest Points) Warm Up
To execute the **icp warm up script**, do:
```bash
cd src/icp_warm_up/
python3 test_icp.py
```
Which should process all Warm Up **Practice Datasets (4 datasests each, for `drill` and `liq_container`, a total of 8 datasets)**

### IMU Odometry
```bash
python3 src/odometry.py dataset=<20 or 21>
```

### Scan Matching
```bash
python3 src/scan_matching.py dataset=<20 or 21>
```

### Pose-Graph Optimization
```bash
python3 src/pose_graph_optimization.py dataset=<20 or 21>
```

### Mapping (did not finished)
```bash
python3 src/mapping.py dataset=<20 or 21> odom_str=lc2
```

### Plotting
```bash
python3 src/plotting.py
```

## Results
Check `[ECE 276A] PR2_Report - Winston Chou.pdf` and `figure/`

## Acknowledgments
This project is part of the **ECE 276A** course at **UC San Diego**, which is inspired by the teaching and research of various professors in the field of robotics and estimation