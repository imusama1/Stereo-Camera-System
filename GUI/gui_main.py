import sys
import os
import yaml
import cv2
import numpy as np
import open3d as o3d
import glob
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton, 
                            QLabel, QFileDialog, QSplitter, QDialog, QGridLayout,
                            QScrollArea, QSlider, QHBoxLayout, QSizePolicy)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage

# WLS Filter Constants
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

def initialize_folders():
    """Create necessary folders if they don't exist"""
    required_folders = [
        "disparity_map",
        "rectified_images",
    ]
    
    for folder in required_folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Created folder: {folder}")

# Initialize folders at startup
initialize_folders()

def resize_images(l, r, w=TARGET_WIDTH, h=TARGET_HEIGHT):
    """
    Resizes the input images to a fixed size.
    """
    return (cv2.resize(l, (w, h), interpolation=cv2.INTER_AREA),
            cv2.resize(r, (w, h), interpolation=cv2.INTER_AREA))

def compute_wls_filtered_disparity(L, R, wls_lambda=10000, wls_sigma=1.2):
    """
    Compute disparity map using StereoSGBM followed by WLS filtering.
    """
    L, R = resize_images(L, R)
    
    # Convert to grayscale
    if len(L.shape) == 3:
        L_gray = cv2.cvtColor(L, cv2.COLOR_BGR2GRAY)
        R_gray = cv2.cvtColor(R, cv2.COLOR_BGR2GRAY)
    else:
        L_gray = L
        R_gray = R

    # Create stereo matcher
    left_matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=16 * 6,
        blockSize=11,
        P1=8 * 3 * 11**2,
        P2=32 * 3 * 11**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    
    # Create right matcher using the left matcher
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    # Compute left and right disparity maps
    left_disp = left_matcher.compute(L_gray, R_gray).astype(np.float32) / 16.0
    right_disp = right_matcher.compute(R_gray, L_gray).astype(np.float32) / 16.0

    # Apply WLS filtering
    wls = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
    wls.setLambda(wls_lambda)  # Controls the smoothness of disparity
    wls.setSigmaColor(wls_sigma)  # Controls color preservation during filtering
    
    # Filter the disparity map
    filtered = wls.filter(left_disp, L, None, right_disp)
    filtered = np.nan_to_num(filtered, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalize disparity map for proper visualization
    disp_norm = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # Apply color map (heatmap) for visualization
    disp_col = cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)

    # Save the filtered disparity map
    os.makedirs("disparity_map", exist_ok=True)
    cv2.imwrite("disparity_map/disparity_filtered_wls.png", disp_col)

    return disp_col

class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('3D Reconstruction')
        
        # Get screen size and set window size to 80% of screen
        screen = QApplication.primaryScreen().geometry()
        width = int(screen.width() * 0.8)
        height = int(screen.height() * 0.8)
        x = int((screen.width() - width) / 2)
        y = int((screen.height() - height) / 2)
        self.setGeometry(x, y, width, height)
        
        self.setStyleSheet(self.get_stylesheet())

        # Main splitter (Left 300px fixed - Right remaining)
        self.splitter = QSplitter(Qt.Horizontal)
        
        # Left Panel (Fixed width)
        self.left_panel = QWidget()
        self.setup_left_panel()
        
        # Right Panel
        self.right_panel = QWidget()
        self.setup_right_panel()

        self.splitter.addWidget(self.left_panel)
        self.splitter.addWidget(self.right_panel)
        self.splitter.setSizes([300, width - 300])

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.splitter)
        self.setLayout(main_layout)

    def setup_left_panel(self):
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(10)  # Add spacing between widgets
        self.left_panel.setStyleSheet("background-color: #222;")
        self.left_panel.setFixedWidth(300)

        # Add a scroll area for the entire left panel
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background-color: #222; border: none;")
        
        # Container for all content
        content_widget = QWidget()
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(10)

        # Calibration and Rectification buttons
        self.calibration_button = QPushButton('Calibration')
        self.calibration_button.setFixedSize(200, 50)
        self.calibration_button.clicked.connect(self.show_calibration_window)
        self.calibration_button.setStyleSheet("background-color: #04BAED; color: white;")
        content_layout.addWidget(self.calibration_button)

        self.rectification_button = QPushButton('Rectification')
        self.rectification_button.setFixedSize(200, 50)
        self.rectification_button.clicked.connect(self.show_rectification_window)
        self.rectification_button.setStyleSheet("background-color: #04BAED; color: white;")
        content_layout.addWidget(self.rectification_button)

        # Simple Disparity Section
        disparity_section = QWidget()
        disparity_section.setStyleSheet("background-color: #333; border-radius: 5px; padding: 10px;")
        disparity_layout = QVBoxLayout()
        disparity_layout.setSpacing(5)
        
        # Section title
        title_label = QLabel("Simple Disparity Settings")
        title_label.setStyleSheet("color: white; font-size: 16px; font-weight: bold; padding: 5px;")
        disparity_layout.addWidget(title_label)
        
        # SGBM parameters with new default values
        self.sgbm_params = {
            'minDisparity': (0, 0, 50, 1),
            'numDisparities': (128, 16, 256, 16),
            'blockSize': (5, 1, 15, 2),
            'P1': (600, 0, 2000, 1),  # 8 * 3 * 5**2 = 600
            'P2': (2400, 0, 5000, 1), # 32 * 3 * 5**2 = 2400
            'disp12MaxDiff': (1, 0, 20, 1),
            'uniquenessRatio': (3, 0, 50, 1),
            'speckleWindowSize': (10, 0, 100, 1),
            'speckleRange': (5, 0, 50, 1),
            'preFilterCap': (2, 0, 50, 1)
        }
        
        # Create sliders with compact layout
        self.sliders = {}
        for param, (default, min_val, max_val, step) in self.sgbm_params.items():
            container = QWidget()
            container.setStyleSheet("background-color: #444; border-radius: 5px; margin: 2px;")
            layout = QVBoxLayout()
            layout.setContentsMargins(5, 2, 5, 2)
            layout.setSpacing(2)
            
            label = QLabel(f"{param}: {default}")
            label.setStyleSheet("color: white; font-size: 12px;")
            slider = QSlider(Qt.Horizontal)
            slider.setFixedHeight(20)
            slider.setRange(min_val, max_val)
            slider.setValue(default)
            slider.setSingleStep(step)
            slider.setPageStep(0)  # Disable scroll wheel/hover changes
            slider.valueChanged.connect(lambda v, lbl=label, p=param: self.update_slider_label(v, lbl, p))
            
            layout.addWidget(label)
            layout.addWidget(slider)
            container.setLayout(layout)
            disparity_layout.addWidget(container)
            self.sliders[param] = (slider, label)

        # Calculate Disparity button
        self.calc_disparity_button = QPushButton('Calculate Disparity')
        self.calc_disparity_button.setFixedSize(200, 50)
        self.calc_disparity_button.clicked.connect(self.calculate_disparity)
        self.calc_disparity_button.setStyleSheet("background-color: #04BAED; color: white;")
        disparity_layout.addWidget(self.calc_disparity_button)

        disparity_section.setLayout(disparity_layout)
        content_layout.addWidget(disparity_section)

        # WLS Filter Section
        wls_section = QWidget()
        wls_section.setStyleSheet("background-color: #333; border-radius: 5px; padding: 10px;")
        wls_layout = QVBoxLayout()
        wls_layout.setSpacing(5)
        
        # Section title
        wls_title = QLabel("WLS Filter Settings")
        wls_title.setStyleSheet("color: white; font-size: 16px; font-weight: bold; padding: 5px;")
        wls_layout.addWidget(wls_title)
        
        # WLS Disparity parameters (same as original implementation)
        self.wls_disparity_params = {
            'minDisparity': (0, 0, 50, 1),
            'numDisparities': (96, 16, 256, 16),  # 16*6 = 96
            'blockSize': (11, 1, 15, 2),
            'P1': (2904, 0, 5000, 1),  # 8*3*11**2 = 2904
            'P2': (11616, 0, 15000, 1), # 32*3*11**2 = 11616
            'disp12MaxDiff': (1, 0, 20, 1),
            'uniquenessRatio': (10, 0, 50, 1),
            'speckleWindowSize': (100, 0, 200, 1),
            'speckleRange': (32, 0, 50, 1)
        }

        # Create WLS disparity sliders
        self.wls_disparity_sliders = {}
        for param, (default, min_val, max_val, step) in self.wls_disparity_params.items():
            container = QWidget()
            container.setStyleSheet("background-color: #444; border-radius: 5px; margin: 2px;")
            layout = QVBoxLayout()
            layout.setContentsMargins(5, 2, 5, 2)
            layout.setSpacing(2)
            
            label = QLabel(f"{param}: {default}")
            label.setStyleSheet("color: white; font-size: 12px;")
            slider = QSlider(Qt.Horizontal)
            slider.setFixedHeight(20)
            slider.setRange(min_val, max_val)
            slider.setValue(default)
            slider.setSingleStep(step)
            slider.setPageStep(0)  # Disable scroll wheel/hover changes
            slider.valueChanged.connect(lambda v, lbl=label, p=param: self.update_wls_disparity_label(v, lbl, p))
            
            layout.addWidget(label)
            layout.addWidget(slider)
            container.setLayout(layout)
            wls_layout.addWidget(container)
            self.wls_disparity_sliders[param] = (slider, label)

        # WLS Filter parameters
        wls_filter_title = QLabel("WLS Filter Parameters")
        wls_filter_title.setStyleSheet("color: white; font-size: 14px; font-weight: bold; padding: 5px;")
        wls_layout.addWidget(wls_filter_title)

        self.wls_params = {
            'lambda': (10000, 0, 20000, 100),
            'sigma': (1.2, 0.0, 5.0, 0.1)
        }
        
        # Create WLS filter parameter sliders
        self.wls_sliders = {}
        for param, (default, min_val, max_val, step) in self.wls_params.items():
            container = QWidget()
            container.setStyleSheet("background-color: #444; border-radius: 5px; margin: 2px;")
            layout = QVBoxLayout()
            layout.setContentsMargins(5, 2, 5, 2)
            layout.setSpacing(2)
            
            label = QLabel(f"{param}: {default}")
            label.setStyleSheet("color: white; font-size: 12px;")
            slider = QSlider(Qt.Horizontal)
            slider.setFixedHeight(20)
            slider.setRange(int(min_val*10) if param == 'sigma' else min_val, 
                        int(max_val*10) if param == 'sigma' else max_val)
            slider.setValue(int(default*10) if param == 'sigma' else default)
            slider.setSingleStep(int(step * 10) if param == 'sigma' else step)
            slider.setPageStep(0)  # Disable scroll wheel/hover changes
            slider.valueChanged.connect(lambda v, lbl=label, p=param: self.update_wls_slider_label(v, lbl, p))
            
            layout.addWidget(label)
            layout.addWidget(slider)
            container.setLayout(layout)
            wls_layout.addWidget(container)
            self.wls_sliders[param] = (slider, label)

        # Apply WLS Filter button
        self.apply_wls_button = QPushButton('Apply WLS Filter')
        self.apply_wls_button.setFixedSize(200, 50)
        self.apply_wls_button.clicked.connect(self.calculate_wls_disparity)
        self.apply_wls_button.setStyleSheet("background-color: #04BAED; color: white;")
        wls_layout.addWidget(self.apply_wls_button)

        wls_section.setLayout(wls_layout)
        content_layout.addWidget(wls_section)

        # 3D Reconstruction button
        self.reconstruction_button = QPushButton('3D Reconstruction')
        self.reconstruction_button.setFixedSize(200, 50)
        self.reconstruction_button.clicked.connect(self.show_3d_reconstruction)
        self.reconstruction_button.setStyleSheet("background-color: #04BAED; color: white;")
        content_layout.addWidget(self.reconstruction_button)

        # Feature Matching button
        self.feature_matching_button = QPushButton('Feature Matching')
        self.feature_matching_button.setFixedSize(200, 50)
        self.feature_matching_button.clicked.connect(self.show_feature_matching)
        self.feature_matching_button.setStyleSheet("background-color: #04BAED; color: white;")
        content_layout.addWidget(self.feature_matching_button)

        # Sparse 3D Reconstruction button
        self.sparse_reconstruction_button = QPushButton('Sparse 3D Reconstruction')
        self.sparse_reconstruction_button.setFixedSize(200, 50)
        self.sparse_reconstruction_button.clicked.connect(self.show_sparse_reconstruction)
        self.sparse_reconstruction_button.setStyleSheet("background-color: #04BAED; color: white;")
        content_layout.addWidget(self.sparse_reconstruction_button)

        # Set the content layout
        content_widget.setLayout(content_layout)
        scroll.setWidget(content_widget)
        left_layout.addWidget(scroll)
        self.left_panel.setLayout(left_layout)

    def setup_right_panel(self):
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(10, 10, 10, 10)
        self.right_panel.setStyleSheet("background-color: #111;")

        self.result_container = QWidget()
        self.result_layout = QGridLayout()
        self.result_container.setLayout(self.result_layout)
        
        right_layout.addWidget(self.result_container)
        self.right_panel.setLayout(right_layout)

    def update_slider_label(self, value, label, param):
        label.setText(f"{param}: {value}")

    def update_wls_slider_label(self, value, label, param):
        if param == 'sigma':
            value = value / 10.0
        label.setText(f"{param}: {value:.1f}" if param == 'sigma' else f"{param}: {value}")

    def update_wls_disparity_label(self, value, label, param):
        label.setText(f"{param}: {value}")

    def calculate_disparity(self):
        try:
            # Get current parameters
            params = {param: slider[0].value() for param, slider in self.sliders.items()}
            
            # Force numDisparities to be divisible by 16
            params['numDisparities'] = max(16, (params['numDisparities'] // 16) * 16)
            
            # Force blockSize to be odd number
            params['blockSize'] = max(1, params['blockSize'] | 1)

            # Always load fresh images from disk
            left_img = cv2.imread("rectified_images/left_rectified.png")
            right_img = cv2.imread("rectified_images/right_rectified.png")
            
            if left_img is None or right_img is None:
                raise FileNotFoundError("Perform rectification first!")

            # Create fresh copies to ensure we're not modifying existing images
            left_img = left_img.copy()
            right_img = right_img.copy()

            # Verify image sizes match calibration data
            with open('cal/stereo__calibration.yaml', 'r') as f:
                calibration_data = yaml.load(f, Loader=yaml.FullLoader)
            expected_size = tuple(calibration_data['image_size'])
            
            # Resize images if necessary
            if left_img.shape[1::-1] != expected_size:
                left_img = cv2.resize(left_img, expected_size)
            if right_img.shape[1::-1] != expected_size:
                right_img = cv2.resize(right_img, expected_size)

            # Convert to grayscale fresh each time
            gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

            # Create new stereo matcher with current parameters
            stereo = cv2.StereoSGBM_create(
                minDisparity=params['minDisparity'],
                numDisparities=params['numDisparities'],
                blockSize=params['blockSize'],
                P1=params['P1'],
                P2=params['P2'],
                disp12MaxDiff=params['disp12MaxDiff'],
                uniquenessRatio=params['uniquenessRatio'],
                speckleWindowSize=params['speckleWindowSize'],
                speckleRange=params['speckleRange'],
                preFilterCap=params['preFilterCap'],
                mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
            )

            # Compute fresh disparity map from scratch
            disp = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
            
            # Clear any previous cached disparity result
            if hasattr(self, '_last_disparity'):
                del self._last_disparity
            
            # Create fresh normalized disparity map
            disp = np.clip(disp, 0, 255)
            disp_norm = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX)
            disparity_img = cv2.applyColorMap(disp_norm.astype(np.uint8), cv2.COLORMAP_JET)

            # Display the fresh result
            self.display_disparity(disparity_img)

        except Exception as e:
            self.show_error(str(e))

    def calculate_wls_disparity(self):
        try:
            # Get current WLS disparity parameters
            disparity_params = {param: slider[0].value() for param, slider in self.wls_disparity_sliders.items()}
            
            # Get current WLS filter parameters
            wls_lambda = self.wls_sliders['lambda'][0].value()
            wls_sigma = self.wls_sliders['sigma'][0].value() / 10.0

            # Load fresh images
            left_img = cv2.imread("rectified_images/left_rectified.png")
            right_img = cv2.imread("rectified_images/right_rectified.png")
            
            if left_img is None or right_img is None:
                raise FileNotFoundError("Perform rectification first!")

            # Create fresh copies
            left_img = left_img.copy()
            right_img = right_img.copy()

            # Resize images
            left_img, right_img = resize_images(left_img, right_img)
            
            # Convert to grayscale
            if len(left_img.shape) == 3:
                gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
                gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
            else:
                gray_left = left_img
                gray_right = right_img

            # Create stereo matcher with current parameters
            left_matcher = cv2.StereoSGBM_create(
                minDisparity=disparity_params['minDisparity'],
                numDisparities=disparity_params['numDisparities'],
                blockSize=disparity_params['blockSize'],
                P1=disparity_params['P1'],
                P2=disparity_params['P2'],
                disp12MaxDiff=disparity_params['disp12MaxDiff'],
                uniquenessRatio=disparity_params['uniquenessRatio'],
                speckleWindowSize=disparity_params['speckleWindowSize'],
                speckleRange=disparity_params['speckleRange'],
                mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
            )

            # Create right matcher
            right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

            # Compute disparity maps
            left_disp = left_matcher.compute(gray_left, gray_right).astype(np.float32) / 16.0
            right_disp = right_matcher.compute(gray_right, gray_left).astype(np.float32) / 16.0

            # Apply WLS filtering with current parameters
            wls = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
            wls.setLambda(wls_lambda)
            wls.setSigmaColor(wls_sigma)
            
            # Filter disparity map
            filtered = wls.filter(left_disp, left_img, None, right_disp)
            filtered = np.nan_to_num(filtered, nan=0.0, posinf=0.0, neginf=0.0)

            # Store the filtered disparity for 3D reconstruction
            self.last_wls_disparity = filtered.copy()

            # Create visualization
            disp_norm = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            disparity_img = cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)

            # Save and display result
            os.makedirs("disparity_map", exist_ok=True)
            cv2.imwrite("disparity_map/disparity_filtered_wls.png", disparity_img)
            self.display_disparity(disparity_img)

        except Exception as e:
            self.show_error(str(e))

    def show_3d_reconstruction(self):
        try:
            # Get the last WLS disparity result
            if not hasattr(self, 'last_wls_disparity'):
                raise Exception("Please run WLS Filter first to generate disparity map")

            disp = self.last_wls_disparity.astype(np.float32)
            print(f"Disparity: min={disp.min():.3f}, max={disp.max():.3f}, mean={np.nanmean(disp):.3f}")
            print(f"Pixels with disp>0: {np.count_nonzero(disp>0)} / {disp.size}")

            # Load Q from YAML
            with open('cal/stereo__calibration.yaml', 'r') as f:
                calib = yaml.load(f, Loader=yaml.FullLoader)
            Q = np.array(calib['Q'], dtype=np.float32)

            # Reproject to 3D
            points3d = cv2.reprojectImageTo3D(disp, Q)

            # Mask: keep disp > 5th percentile of positive values
            pos = disp[disp>0]
            thr = np.percentile(pos, 5) if pos.size else 0
            mask = disp > thr

            # Extract & color
            left_img = cv2.imread("rectified_images/left_rectified.png")
            img_color = cv2.resize(left_img, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)
            pts = points3d[mask]
            cols = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)[mask]

            # Downsample ~100k pts for speed
            step = max(1, len(pts)//100000)
            pts_ds = pts[::step]
            cols_ds = cols[::step]

            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts_ds)
            pcd.colors = o3d.utility.Vector3dVector(cols_ds.astype(np.float64)/255.0)

            # Flip 180° about X axis
            R = np.array([[1, 0, 0],
                         [0, -1, 0],
                         [0, 0, -1]], dtype=np.float64)
            pcd.rotate(R, center=(0,0,0))

            # Show in separate window since Open3D visualization can't be embedded in Qt
            o3d.visualization.draw_geometries([pcd])

        except Exception as e:
            self.show_error(str(e))

    def show_feature_matching(self):
        try:
            # Load stereo images
            img_left = cv2.imread("rectified_images/left_rectified.png")
            img_right = cv2.imread("rectified_images/right_rectified.png")
            
            if img_left is None or img_right is None:
                raise FileNotFoundError("Perform rectification first!")

            # Initialize ORB detector
            orb = cv2.ORB_create(5000)

            # Detect keypoints and compute descriptors
            kp1, des1 = orb.detectAndCompute(img_left, None)
            kp2, des2 = orb.detectAndCompute(img_right, None)

            print(f"Keypoints detected - Left: {len(kp1)}, Right: {len(kp2)}")

            # Match descriptors using Brute-Force Matcher
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)

            # Visualize top 100 matches
            img_matches = cv2.drawMatches(img_left, kp1, img_right, kp2, matches[:100], None, flags=2)

            # Store matched points for sparse reconstruction
            self.pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
            self.pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

            # Display the matches
            self.display_matches(img_matches)

        except Exception as e:
            self.show_error(str(e))

    def display_matches(self, img_matches):
        self.clear_results()
        h, w, ch = img_matches.shape
        bytes_per_line = ch * w
        q_img = QImage(img_matches.data, w, h, bytes_per_line, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(q_img).scaled(1000, 600, Qt.KeepAspectRatio)
        
        label = QLabel()
        label.setPixmap(pixmap)
        self.result_layout.addWidget(label, 0, 0)

    def show_sparse_reconstruction(self):
        try:
            if not hasattr(self, 'pts1') or not hasattr(self, 'pts2'):
                raise Exception("Please run Feature Matching first!")

            # Load calibration data
            with open('cal/stereo__calibration.yaml', 'r') as f:
                calib = yaml.safe_load(f)

            # Convert lists to numpy arrays
            for key in ['K_left', 'K_right', 'P1', 'P2', 'R', 'T']:
                calib[key] = np.array(calib[key])

            # Compute Fundamental Matrix and filter inliers
            F, inliers = cv2.findFundamentalMat(self.pts1, self.pts2,
                                              method=cv2.FM_RANSAC,
                                              ransacReprojThreshold=3.0,
                                              confidence=0.999)

            pts1_filtered = self.pts1[inliers.ravel() == 1]
            pts2_filtered = self.pts2[inliers.ravel() == 1]

            # Get projection matrices
            P1 = calib['P1']
            P2 = calib['P2']

            # Safe triangulation
            points3D, valid_mask = self.safe_triangulate(P1, P2, pts1_filtered, pts2_filtered)

            # Create colored point cloud
            img_left = cv2.imread("rectified_images/left_rectified.png")
            valid_colors = [img_left[int(y), int(x)][::-1]/255.0 
                          for (x,y), valid in zip(pts1_filtered, valid_mask) if valid]

            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points3D)
            pcd.colors = o3d.utility.Vector3dVector(valid_colors)

            # Coordinate system correction
            pcd.transform([[1,0,0,0], [0,-1,0,0], [0,0,-1,0], [0,0,0,1]])

            print("✅ Point cloud created with colors from left image")
            o3d.visualization.draw_geometries([pcd],
                                           window_name="Sparse 3D Reconstruction",
                                           width=1280,
                                           height=720)

        except Exception as e:
            self.show_error(str(e))

    def safe_triangulate(self, P1, P2, pts1, pts2, max_depth=10.0):
        """Improved triangulation with depth constraints."""
        pts4D = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)

        # Normalize homogeneous coordinates
        valid_mask = (np.abs(pts4D[3]) > 1e-6)
        pts3D = pts4D[:3] / pts4D[3]

        # Apply physical constraints
        z_mask = (pts3D[2] > 0.3) & (pts3D[2] < max_depth)  # 0.3m to 10m range
        x_mask = (np.abs(pts3D[0]) < 5.0)  # ±5m horizontal
        y_mask = (np.abs(pts3D[1]) < 5.0)  # ±5m vertical

        final_mask = valid_mask & z_mask & x_mask & y_mask

        return pts3D[:, final_mask].T, final_mask

    def display_disparity(self, disparity_img):
        self.clear_results()
        h, w, ch = disparity_img.shape
        bytes_per_line = ch * w
        q_img = QImage(disparity_img.data, w, h, bytes_per_line, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(q_img).scaled(1000, 600, Qt.KeepAspectRatio)
        
        label = QLabel()
        label.setPixmap(pixmap)
        self.result_layout.addWidget(label, 0, 0)

    def show_error(self, message):
        error_dialog = QDialog(self)
        error_dialog.setWindowTitle("Error")
        error_dialog.setStyleSheet(self.styleSheet())
        layout = QVBoxLayout()
        error_label = QLabel(message)
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(error_dialog.close)
        layout.addWidget(error_label)
        layout.addWidget(ok_button)
        error_dialog.setLayout(layout)
        error_dialog.exec_()

    def show_calibration_window(self):
        self.calibration_dialog = CalibrationDialog(self)
        self.calibration_dialog.exec_()

    def show_rectification_window(self):
        self.rectification_dialog = RectificationDialog(self)
        self.rectification_dialog.exec_()

    def display_calibration_results(self, error_left, error_right):
        self.clear_results()
        result_label = QLabel(f"Calibration Complete!\n\nLeft Camera Error: {error_left}")
        result_label.setAlignment(Qt.AlignCenter)
        result_label.setStyleSheet("font-size: 16px; color: white;")
        self.result_layout.addWidget(result_label, 0, 0)

    def display_rectification_results(self, output_folder):
        self.clear_results()
        
        images = [
            ("left_rectified.png", "Left Rectified"),
            ("right_rectified.png", "Right Rectified"),
            ("left_rectified_with_lines.png", "Left with Epipolar Lines"),
            ("right_rectified_with_lines.png", "Right with Epipolar Lines")
        ]

        row, col = 0, 0
        for img_file, title in images:
            img_path = os.path.join(output_folder, img_file)
            if os.path.exists(img_path):
                pixmap = QPixmap(img_path).scaled(400, 300, Qt.KeepAspectRatio)
                label = QLabel()
                label.setPixmap(pixmap)
                title_label = QLabel(title)
                title_label.setAlignment(Qt.AlignCenter)
                title_label.setStyleSheet("color: white; font-size: 14px;")
                
                self.result_layout.addWidget(title_label, row, col)
                self.result_layout.addWidget(label, row+1, col)
                col += 1
                if col > 1:
                    col = 0
                    row += 2

    def clear_results(self):
        for i in reversed(range(self.result_layout.count())): 
            self.result_layout.itemAt(i).widget().deleteLater()

    def get_stylesheet(self):
        return """
        QWidget {
            background-color: #121212;
            color: white;
            font-family: Arial, sans-serif;
        }
        QPushButton {
            background-color: #1f1f1f;
            border: none;
            color: white;
            padding: 12px;
            font-size: 16px;
            border-radius: 5px;
            margin: 5px;
        }
        QPushButton:hover {
            background-color: #333;
        }
        QLabel {
            font-size: 14px;
            margin-bottom: 10px;
        }
        """

# CalibrationDialog code (updated to read calibration data from file)
class CalibrationDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Calibration - Select Camera Folders")
        self.setGeometry(200, 200, 400, 200)
        self.setStyleSheet(parent.styleSheet())

        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)

        # Instruction label
        self.instructions_label = QLabel("Please upload Left and Right Camera Images folders.")
        layout.addWidget(self.instructions_label)

        # Folder selection buttons
        self.left_folder_button = QPushButton("Left Camera Images")
        self.left_folder_button.clicked.connect(self.upload_left_folder)
        layout.addWidget(self.left_folder_button)

        self.right_folder_button = QPushButton("Right Camera Images")
        self.right_folder_button.clicked.connect(self.upload_right_folder)
        layout.addWidget(self.right_folder_button)

        # Calibrate button
        self.calibrate_button = QPushButton("Calibrate")
        self.calibrate_button.clicked.connect(self.calibrate)
        self.calibrate_button.setEnabled(False)
        layout.addWidget(self.calibrate_button)

        self.setLayout(layout)

    def upload_left_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Left Camera Images Folder")
        if folder_path:
            self.left_folder_path = folder_path
            self.left_folder_button.setText(f"Left: {os.path.basename(folder_path)}")
            self.check_folders_selected()

    def upload_right_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Right Camera Images Folder")
        if folder_path:
            self.right_folder_path = folder_path
            self.right_folder_button.setText(f"Right: {os.path.basename(folder_path)}")
            self.check_folders_selected()

    def check_folders_selected(self):
        if hasattr(self, 'left_folder_path') and hasattr(self, 'right_folder_path'):
            self.calibrate_button.setEnabled(True)

    def calibrate(self):
        # Load the calibration results from the calibration file
        calibration_file = 'cal/stereo__calibration.yaml'

        if os.path.exists(calibration_file):
            with open(calibration_file, 'r') as f:
                calibration_data = yaml.load(f, Loader=yaml.FullLoader)

            # Read calibration errors from the file
            error_left = f"{calibration_data.get('error_left', 'N/A'):.4f}"
            error_right = f"{calibration_data.get('error_right', 'N/A'):.4f}"
            stereo_error = f"{calibration_data.get('stereo_error', 'N/A'):.4f}"

            # Display comprehensive calibration results
            results = (f"Left Camera Mean Reprojection Error: {error_left}\n"
                      f"Right Camera Mean Reprojection Error: {error_right}")
                    #   f"Stereo Calibration Error: {stereo_error} pixels")
            
            self.parent().display_calibration_results(results, "")
        else:
            self.parent().display_calibration_results("Error: Calibration file not found!", 
                                                    f"Expected at: {os.path.abspath(calibration_file)}")
        
        self.close()

class RectificationDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Image Rectification")
        self.setGeometry(200, 200, 400, 200)
        self.setStyleSheet(parent.styleSheet())

        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)

        self.instructions_label = QLabel("Select left and right images for rectification")
        layout.addWidget(self.instructions_label)

        self.left_image_button = QPushButton("Select Left Image")
        self.left_image_button.clicked.connect(self.select_left_image)
        layout.addWidget(self.left_image_button)

        self.right_image_button = QPushButton("Select Right Image")
        self.right_image_button.clicked.connect(self.select_right_image)
        layout.addWidget(self.right_image_button)

        self.rectify_button = QPushButton("Rectify")
        self.rectify_button.clicked.connect(self.rectify_images)
        self.rectify_button.setEnabled(False)
        layout.addWidget(self.rectify_button)

        self.setLayout(layout)

    def select_left_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Left Image", "", "Images (*.png *.jpg *.jpeg)")
        if file_path:
            self.left_image_path = file_path
            self.left_image_button.setText(f"Left: {os.path.basename(file_path)}")
            self.check_images_selected()

    def select_right_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Right Image", "", "Images (*.png *.jpg *.jpeg)")
        if file_path:
            self.right_image_path = file_path
            self.right_image_button.setText(f"Right: {os.path.basename(file_path)}")
            self.check_images_selected()

    def check_images_selected(self):
        if hasattr(self, 'left_image_path') and hasattr(self, 'right_image_path'):
            self.rectify_button.setEnabled(True)

    def rectify_images(self):
        try:
            output_folder = "rectified_images"
            os.makedirs(output_folder, exist_ok=True)

            # Load calibration data
            with open('cal/stereo__calibration.yaml', 'r') as f:
                calibration_data = yaml.load(f, Loader=yaml.FullLoader)

            K_left = np.array(calibration_data['K_left'])
            D_left = np.array(calibration_data['D_left'])
            K_right = np.array(calibration_data['K_right'])
            D_right = np.array(calibration_data['D_right'])
            image_size = tuple(calibration_data['image_size'])

            # Load images
            img_left = cv2.imread(self.left_image_path)
            img_right = cv2.imread(self.right_image_path)

            # Rectification process
            img_left_undistorted = cv2.undistort(img_left, K_left, D_left)
            img_right_undistorted = cv2.undistort(img_right, K_right, D_right)

            R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
                K_left, D_left, K_right, D_right, image_size,
                np.eye(3), np.array([[-0.1], [0.0], [0.0]]),
                flags=cv2.CALIB_ZERO_DISPARITY, alpha=1.0
            )

            map_left_x, map_left_y = cv2.initUndistortRectifyMap(K_left, D_left, R1, P1, image_size, cv2.CV_32FC1)
            map_right_x, map_right_y = cv2.initUndistortRectifyMap(K_right, D_right, R2, P2, image_size, cv2.CV_32FC1)

            img_left_rectified = cv2.remap(img_left_undistorted, map_left_x, map_left_y, cv2.INTER_LINEAR)
            img_right_rectified = cv2.remap(img_right_undistorted, map_right_x, map_right_y, cv2.INTER_LINEAR)

            # Save images
            cv2.imwrite(f"{output_folder}/left_rectified.png", img_left_rectified)
            cv2.imwrite(f"{output_folder}/right_rectified.png", img_right_rectified)

            # Create and save images with epipolar lines
            line_interval = 50
            for img, name in [(img_left_rectified, "left"), (img_right_rectified, "right")]:
                img_with_lines = img.copy()
                for y in range(0, img.shape[0], line_interval):
                    cv2.line(img_with_lines, (0, y), (img.shape[1], y), (150, 255, 50), 1)
                cv2.imwrite(f"{output_folder}/{name}_rectified_with_lines.png", img_with_lines)

            self.parent().display_rectification_results(output_folder)
            self.close()

        except Exception as e:
            self.parent().display_calibration_results('Rectification Error', str(e))
            self.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())
