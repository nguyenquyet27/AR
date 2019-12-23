import os
import numpy as np

from reference_plane import ReferencePlane
from objloader_simple import OBJ
MIN_MATCHES = 120


# TODO: estimate this
camera_intrinsic = np.array(
    [[600, 0, 320],
     [0, 600, 240],
     [0, 0, 1]]
)


joker = ReferencePlane('template/joker.jpg')
# TODO cai swapyxz lam gi vay?
_3d_fox = OBJ('models/fox.obj', swapyz=True)
