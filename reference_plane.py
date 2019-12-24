import cv2
import os

from process_func import image_proc
import config


class ReferencePlane(object):

    def __init__(self, plane_path):
        print('[INFO]Make reference surface ...')
        self.orb = cv2.ORB_create()
        self.sift = cv2.xfeatures2d.SIFT_create()
        # TODO: using more planar
        self.ref_plane = cv2.imread(os.path.join(os.getcwd(), plane_path))

        self.image_ref = image_proc(self.ref_plane, 1)
        self.height, self.width = self.image_ref.shape

        self.keypoints, self.descriptors = self.orb.detectAndCompute(
            self.image_ref, None)
        # self.keypoints, self.descriptors = self.sift.detectAndCompute(
        #     self.image_ref, None)

    def get_keypoints(self):
        return self.keypoints

    def get_descriptors(self):
        return self.descriptors
