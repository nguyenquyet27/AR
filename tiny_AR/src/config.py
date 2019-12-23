import os
import cv2
import numpy as np
from django.conf import settings

from .obj import OBJModel
from .preprocess import image_proc

calibration_matrix = np.array(
    [[600, 0, 640],
     [0, 600, 360],
     [0, 0, 1]])

MIN_MATCHES = 50

model = cv2.imread(os.path.join(settings.BASE_DIR, 'src/ref/joker.jpg'))
model_height, model_width = model.shape
points = np.float32(
    [[0, 0],
     [0, model_height - 1],
     [model_width - 1, model_height - 1],
     [model_width - 1, 0]]
).reshape(-1, 1, 2)
# print('widht: {}, height: {}'.format(
#     model_width, model_height))
img = image_proc(model, 0.5)

orb = cv2.ORB_create()
kp_model, des_model = orb.detectAndCompute(img, None)

obj = OBJModel(os.path.join(settings.BASE_DIR,
                            'src/models/11571_Gingerbread_cookie_male_V2_l2.obj'), swapyz=True)
