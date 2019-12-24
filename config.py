import os
import numpy as np
import cv2

from reference_plane import ReferencePlane
from objloader_simple import OBJ

MIN_MATCHES = 20


image_plane_width = 640
image_plane_height = 480


# TODO: estimate this
camera_intrinsic = np.array(
    [[900, 0, image_plane_width/2],
     [0, 900, image_plane_height/2],
     [0, 0, 1]]
)
# print(camera_intrinsic)


joker = ReferencePlane('template/pike.jpg')
cv2.imshow('joker', joker.image_ref)
judgement = ReferencePlane('template/judgement.jpg')
bookmark = ReferencePlane('template/bookmark.jpg')

_3d_fox = OBJ('models/fox.obj', swapyz=True)


# ORB + FLANN
FLANN_INDEX_KDTREE = 1
FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH,
                    table_number=6,  # 12
                    key_size=12,     # 20
                    multi_probe_level=1)  # 2
