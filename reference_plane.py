import cv2
import os

from process_func import image_proc
import config


class ReferencePlane(object):

    def __init__(self, plane_path):
        print('[INFO]Make reference surface ...')
        self.orb = cv2.ORB_create()

        # TODO: using more planar
        self.ref_plane = cv2.imread(os.path.join(os.getcwd(), plane_path))
        # Set size of image in dataset if you want. Change here (230,300)
        self.image_ref = image_proc(cv2.resize(self.ref_plane, (230,300)), 1)
        self.height, self.width = self.image_ref.shape

        self.keypoints, self.descriptors = self.orb.detectAndCompute(
            self.image_ref, None)

    def get_keypoints(self):
        return self.keypoints

    def get_descriptors(self):
        return self.descriptors
