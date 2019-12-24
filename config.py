import os
import numpy as np

from reference_plane import ReferencePlane
from objloader_simple import OBJ
MIN_MATCHES = 80


image_plane_width = 640
image_plane_height = 480


# TODO: estimate this
camera_intrinsic = np.array(
    [[800, 0, image_plane_width/2],
     [0, 800, image_plane_height/2],
     [0, 0, 1]]
)
print(camera_intrinsic)


joker = ReferencePlane('template/joker.jpg')
judgement = ReferencePlane('template/judgement.jpg')
bookmark = ReferencePlane('template/bookmark.jpg')

# TODO cai swapyxz lam gi vay?
_3d_fox = OBJ('models/fox.obj', swapyz=True)
