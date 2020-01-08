import os
import numpy as np
import cv2

from reference_plane import ReferencePlane
from objloader import OBJ

MIN_MATCHES = 80


image_plane_width = 720
image_plane_height = 480


# Estimate using ./cam-parameters/compute_calibration_matrix.py
camera_intrinsic = np.array(
    [[900, 0, image_plane_width/2],
     [0, 900, image_plane_height/2],
     [0, 0, 1]]
)


marker = ReferencePlane('template/marker.jpg')
marker2 = ReferencePlane('template/marker2.jpg')

_3d_fox = OBJ('models/fox.obj', swapyz=True)


# ORB + FLANN configuration
FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH,
                    table_number=6,  # 12
                    key_size=12,     # 20
                    multi_probe_level=1)  # 2
