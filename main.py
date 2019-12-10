import os
import cv2 
import math
import argparse
import numpy as np 
from process import *
from model_3d import *
from objloader_simple import *

MIN_MATCHES = 100

def detec_tracking(filemodel):
    homography = None 
    camera_parameters = my_calibration((800,1000))
    # extract feature orb and matching by Brute Force
    orb = cv2.ORB_create() 
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    dir_name = os.getcwd()
    model = cv2.imread(os.path.join(dir_name, filemodel), 0)
    # Compute model keypoints and its descriptors
    kp_model, des_model = orb.detectAndCompute(model, None)
    cap = cv2.VideoCapture(0)
    # Read frame and take bbox
    while True:
        # read the current frame
        ret, frame = cap.read()
        if not ret:
            print("Unable to capture video")
            return 
        kp_frame, des_frame = orb.detectAndCompute(frame, None)
        matches = bf.match(des_model, des_frame)
        matches = sorted(matches, key=lambda x: x.distance)
        if (len(matches) > MIN_MATCHES):
            bbox = compute_homography(kp_model,kp_frame,matches,model)
            #Tracking
            frame = tracking(frame,bbox)
        else:
            print ("Not enough matches found - %d/%d" % (len(matches), MIN_MATCHES))
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    filemodel = 'marker/joker.jpg'
    detec_tracking(filemodel)
    
if __name__ == '__main__':
    main()
