# 📷 Stereo Camera Calibration Toolkit

Welcome to the Stereo Camera Calibration Toolkit! This project is designed to help you calibrate a stereo camera setup using OpenCV in Python. Whether you're working on a robotics project, a 3D reconstruction task, or just exploring computer vision, this notebook should give you a solid foundation.

---

## 🔧 What It Does

- Calibrates individual cameras (left and right) using checkerboard patterns.
- Performs stereo calibration
- Applies stereo rectification to align the views horizontally.
- Saves all important parameters into a clean YAML file for later use.

---

## 🧰 Dependencies

Make sure you’ve got the following packages installed:

```bash
pip install opencv-python numpy pyyaml matplotlib
```


## 📐 Calibration Setup

- **Checkerboard size:** 7x9 inner corners
- **Each square size:** 15mm (0.015m)
- **Image resolution:** 1280x720

Be sure the entire checkerboard is visible and clear in each image. Good lighting and sharp focus go a long way.

---

##  How to Use

1. Collect stereo images of a checkerboard pattern from both cameras.
2. Save the left camera images in the `left/` folder and right camera images in the `right/` folder.
3. Open and run `main_file.ipynb` step by step.
4. Once done, check the `stereo_calibration.yaml` file for the saved parameters.

---

## 📝 What's in the YAML Output

The calibration output file contains:

- `K_left`, `K_right`: Camera intrinsics
- `D_left`, `D_right`: Distortion coefficients
- `R`, `T`: Rotation and translation between cameras
- `R1`, `R2`, `P1`, `P2`, `Q`: Rectification and projection matrices
- Reprojection errors for each calibration step

---

## ✅ Tips

- Use at least 25–30 image pairs for best results.
- Try to capture images from different angles and positions.
- Avoid blurry or overexposed images.

---

## 🤝 Contributing

Found a bug or have a suggestion? Feel free to open an issue or submit a pull request.

---

Thanks for checking this out! Hope it helps in your stereo vision journey. 