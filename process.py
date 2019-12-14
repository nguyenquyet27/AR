import os
import cv2 
import math
import argparse
import numpy as np 
from model_3d import *

def my_calibration(sz):
    """
    Calibration function for the camera (iPhone4) used in this example.
    """
    row,col = sz
    fx = 2555*col/2592
    fy = 2586*row/1936
    K = np.diag([fx,fy,1])
    K[0,2] = 0.5*col
    K[1,2] = 0.5*row
    return K


def compute_homography(kp_model,kp_frame,matches,model):
    # differenciate between source points and destination points
    src_pts = np.float32([kp_model[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    # compute Homography
    homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    # Draw a rectangle that marks the found model in the frame
    h, w = model.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    # project corners into frame
    dst = cv2.perspectiveTransform(pts, homography)
    # take bbox
    boxwidth = np.abs(dst[0][0][0] - dst[2][0][0])
    boxheight = np.abs(dst[0][0][1] - dst[2][0][1])
    bbox = (dst[2][0][0],dst[2][0][1],boxwidth,boxheight)
    return bbox


def projection_matrix(camera_parameters, homography):
    """
    From the camera calibration matrix and the estimated homography
    compute the 3D projection matrix
    """
    # Compute rotation along the x and y axis as well as the translation
    homography = homography * (-1)
    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]
    # normalise vectors
    l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l
    # compute the orthonormal basis
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_3 = np.cross(rot_1, rot_2)
    # finally, compute the 3D projection matrix from the model to the current frame
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T
    return np.dot(camera_parameters, projection)

def tracking(frame,bbox):
    tracker_type = 'KCF'
    tracker = cv2.TrackerKCF_create()
    ret = tracker.init(frame, bbox)
    # Start timer
    timer = cv2.getTickCount()
    # Update tracker
    ret, bbox = tracker.update(frame)
    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
    # Draw bounding box
    if ret:
        # Tracking success
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        frame = cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
        width = np.abs(p1[0]-p2[0])
        height = width = np.abs(p1[1]-p2[1])
        # frame = render(width,height)
        # frame = render(frame, obj, projection, model, False)
    # else :
    #     # Tracking failure
    #     cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
    return frame

def detect_boundingbox(filemodel,frame,MIN_MATCHES):
    homography = None 
    camera_parameters = my_calibration((800,1000))
    # extract feature orb and matching by Brute Force
    orb = cv2.ORB_create() 
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    dir_name = os.getcwd()
    model = cv2.imread(os.path.join(dir_name, filemodel), 0)
    # Compute model keypoints and its descriptors
    kp_model, des_model = orb.detectAndCompute(model, None)
    kp_frame, des_frame = orb.detectAndCompute(frame, None)
    matches = bf.match(des_model, des_frame)
    matches = sorted(matches, key=lambda x: x.distance)
    if (len(matches) > MIN_MATCHES):
        bbox = compute_homography(kp_model,kp_frame,matches,model)
        #Tracking
        frame = tracking(frame,bbox)
    else:
        print ("Not enough matches found - %d/%d" % (len(matches), MIN_MATCHES))
    return frame

