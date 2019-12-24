import cv2
import time
import numpy as np
import process_func as pf
from objloader_simple import *
import math

from pygame.locals import *
from pygame.constants import *
from OpenGL.GL import *
from OpenGL.GLU import *

MIN_MATCHES = 300
def main():
    """
    This functions loads the target surface image,
    """
    homography = None 
    camera_parameters = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])
    # For ORB: NORM_HAMING, NORM_HAMING2
    # detector = cv2.ORB_create()
    # For SIFT: NORM_L1, NORM_L2
    # detector = cv2.xfeatures2d.SIFT_create()
    # For SURF: NORM_L1, NORM_L2
    # minHessian = 400
    # detector = cv2.xfeatures2d_SURF.create(hessianThreshold=minHessian)
    # For Brisk
    detector = cv2.BRISK_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    model = cv2.imread('template/joker.jpg')
    obj = OBJ('models/fox.obj',swapyz=True)
    img1 = pf.image_proc(model, 1)
    kp_model, des_model = detector.detectAndCompute(img1, None)
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        img2 = pf.image_proc(frame, 1)
        # print(frame.shape)
        if not ret:
            print ("Unable to capture video")
            return 
        start_time = time.time()
        kp_frame, des_frame = detector.detectAndCompute(img2, None)
        if (des_frame is None): 
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        matches = bf.match(des_model, des_frame)
        matches = sorted(matches, key=lambda x: x.distance)

        if (len(matches) > MIN_MATCHES):
            src_pts = np.float32([kp_model[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            homography = pf.computeHomography(src_pts ,dst_pts)
            h, w = img1.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, homography)
            # frame = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
            if homography is not None:
                try:
                    # obtain 3D projection matrix from homography matrix and camera parameters
                    projection = pf.projection_matrix(camera_parameters, homography)
                    # project cube or model
                    frame = pf.render(frame, obj, projection, img1, False)
                except:
                    pass
            end_time = time.time()
            print ('total run-time: %f ms' % ((end_time - start_time) * 1000))
            
        else:
            print ("Not enough matches found - %d/%d" % (len(matches), MIN_MATCHES)) 
        
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0

if __name__ == '__main__':
    main()