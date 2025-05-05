Here’s a sample **GitHub README** for your project. It provides an overview of the project, how to install dependencies, run the application, and use its features.

---

# 3D Reconstruction using stereo camera

This project is a Stereo Vision Processing application that uses stereo images to generate disparity maps, perform 3D reconstruction, and display feature matching results. It leverages OpenCV for image processing, Open3D for 3D visualization, and PyQt5 for the graphical user interface (GUI).

## Features

* **Stereo Disparity Map Generation:** Use stereo images to generate disparity maps using the SGBM (Semi-Global Block Matching) algorithm and WLS (Weighted Least Squares) filtering.
* **3D Reconstruction:** Reconstruct a 3D point cloud from the disparity map.
* **Feature Matching:** Match features between stereo images using ORB (Oriented FAST and Rotated BRIEF) detectors.
* **Rectification:** Rectify stereo images to align epipolar lines for easier depth estimation.
* **Calibration:** Use camera calibration data (stored in a YAML file) to perform stereo calibration.

## Requirements

Before running the application, ensure that you have the following dependencies installed:

* Python 3.8 or higher
* PyQt5
* OpenCV
* Open3D
* NumPy
* PyYAML

### Installing Dependencies

If you are using **Conda**, you can create a new environment and install the necessary packages with the following commands:

```bash
conda create -n stereo_vision python=3.8
conda activate stereo_vision
conda install opencv pyqt5 numpy open3d pyyaml
```

Alternatively, you can use `pip` to install the required packages:

```bash
pip install opencv-python PyQt5 numpy open3d pyyaml
```

## Usage

1. **Run the Application:**

   After setting up the environment, run the main application using the following command:

   ```bash
   python stereo_vision_processor.py
   ```

2. **User Interface:**

   The GUI provides several buttons for different functionalities:

   * **Calibration:** Load and calibrate the stereo camera setup using the provided YAML file.
   * **Rectification:** Select left and right images and rectify them for disparity calculation.
   * **Disparity Calculation:** Use the rectified images to compute the disparity map.
   * **Apply WLS Filter:** Apply the WLS filtering to smooth the disparity map.
   * **3D Reconstruction:** Generate and display a 3D point cloud from the disparity map.
   * **Feature Matching:** Match features between the left and right stereo images.
   * **Sparse 3D Reconstruction:** Perform a sparse 3D reconstruction based on feature matches.

3. **Calibration YAML File:**

   The calibration parameters (intrinsic matrix, distortion coefficients, etc.) are stored in a YAML file (`stereo__calibration.yaml`). This file is required for accurate rectification and disparity calculations. You can create this file using standard camera calibration methods (e.g., using a checkerboard pattern and OpenCV functions).

4. **Image Files:**

   Ensure you have a set of stereo images for processing. The images should be stored in the `rectified_images/` folder. Make sure they are named `left_rectified.png` and `right_rectified.png`.

## Project Structure

```
StereoVisionProcessor/
├── stereo_vision_processor.py      # Main script for the application
├── cal/                           # Folder for calibration files (YAML)
│   └── stereo__calibration.yaml   # Calibration data file
├── rectified_images/              # Folder for rectified stereo images
├── disparity_map/                 # Folder for disparity maps
├── requirements.txt               # List of project dependencies (for pip users)
└── README.md                      # Project documentation
```

## How It Works

1. **Disparity Map Calculation:**
   The disparity map is calculated using the StereoSGBM algorithm from OpenCV, followed by WLS filtering to refine the result. This disparity map represents the depth information of the scene, where closer objects have higher disparity values.

2. **3D Reconstruction:**
   The disparity map is reprojected to 3D space using the camera calibration parameters (from the YAML file). This results in a 3D point cloud that can be visualized using Open3D.

3. **Feature Matching:**
   ORB (Oriented FAST and Rotated BRIEF) feature matching is used to identify corresponding points between the stereo images. These matches are then used to perform sparse 3D reconstruction by triangulating the matched points.

## Building the Executable

You can convert this Python application into a standalone `.exe` file using **PyInstaller**.

### Steps:

1. Install **PyInstaller**:

   ```bash
   conda install pyinstaller
   ```

2. Build the executable:

   ```bash
   pyinstaller --onefile --windowed --add-data "cal/stereo__calibration.yaml;cal" --hidden-import cv2 --hidden-import PyQt5 --hidden-import open3d --hidden-import numpy --hidden-import yaml stereo_vision_processor.py
   ```

   This will create an executable file in the `dist/` folder, which you can run on Windows without needing to install Python.


---

This README provides the necessary instructions to set up and use the project, along with the structure of the project and the process of building an executable. Make sure to adjust paths or specific details to match the actual setup and implementation of your project.
