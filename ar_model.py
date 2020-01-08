import cv2
import numpy as np
import config

import process_func as pf


class ARModel(object):
    """
        with each card (hopefully), we make into an object so that we can project 3d model onto it
    """

    def __init__(self, reference_plane, target_plane):
        self.homography = None
        # TODO: other handcrafts feature?
        self.orb = cv2.ORB_create()

        # TODO: other distance formula?
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.flann = cv2.FlannBasedMatcher(config.index_params)

        self.target = target_plane
        self.set_preprocess_target()
        self.keypoints, self.descriptors = self.orb.detectAndCompute(
            self.target_after, None)

    def set_preprocess_target(self):
        self.target_after = pf.image_proc(self.target, 1)

    def get_preprocess_target(self):
        return self.target_after

    def get_keypoints(self):
        return self.keypoints

    def get_descriptors(self):
        return self.descriptors

    def set_matches(self, reference_plane):
        self.matches = self.flann.knnMatch(
            reference_plane.get_descriptors(), self.descriptors, k=2)
        self.matches = [m[0] for m in self.matches if len(
            m) == 2 and m[0].distance < m[1].distance * 0.75]

    def get_matches(self):
        return self.matches

    def set_homography(self, reference_plane):
        """
            set homography for target surface object which transform [X,Y,0,1].tranpose to z[u,v,1].transpose
        """
        ref_kp = reference_plane.get_keypoints()

        src_points = np.float32(
            [ref_kp[m.queryIdx].pt for m in self.matches]).reshape(-1, 1, 2)
        dst_points = np.float32(
            [self.keypoints[m.trainIdx].pt for m in self.matches]).reshape(-1, 1, 2)

        # TODO so 4.0 do lam gi vay?
        H, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 10.0)

        self.homography = H

        return dst_points

    def get_homography(self):
        try:
            return self.homography
        except:
            print("Maybe you hasn't calculate homomatrix?")
