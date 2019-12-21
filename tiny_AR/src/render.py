import cv2
import numpy as np
import math

from .obj import OBJModel
from . import config
from .preprocess import image_proc


class Render(object):

    def __init__(self, frame):
        super().__init__()
        self.image = frame
        self.image_line = None
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self.homography = None
        self.kp = None
        self.des = None
        self.matches = None

    def set_preprocess(self):
        self.image_line = image_proc(self.image, 0.5)

        return self.image_line

    def feature_matching(self):
        self.set_preprocess()
        self.kp, self.des = self.orb.detectAndCompute(self.image_line, None)

        self.matches = self.bf.match(config.des_model, self.des)
        self.matches = sorted(self.matches, key=lambda x: x.distance)

        return self.matches

    def compute_homo(self, src_points, des_points):
        H, mask = cv2.findHomography(src_points, des_points, cv2.RANSAC, 5.0)

        return H

    def estimate_homomatrix(self):
        src_points = np.float32(
            [config.kp_model[m.queryIdx].pt for m in self.matches]).reshape(-1, 1, 2)
        dst_points = np.float32(
            [self.kp[m.trainIdx].pt for m in self.matches]).reshape(-1, 1, 2)

        self.homography = self.compute_homo(src_points, dst_points)

        dst = cv2.perspectiveTransform(config.points, self.homography)
        # print(dst)

        src_points = np.copy(dst_points)

        self.image = cv2.polylines(
            self.image, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
