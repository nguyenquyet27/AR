import cv2
import numpy as np
import process_func as pf
from objloader_simple import *
import math

MIN_MATCHES = 100
def main():
    """
    This functions loads the target surface image,
    """
    homography = None 
    camera_parameters = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    model = cv2.imread('template/joker.jpg')
    obj = OBJ('models/fox.obj',swapyz=True)
    img1 = pf.image_proc(model, 0.5)
    kp_model, des_model = orb.detectAndCompute(img1, None)
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        img2 = pf.image_proc(frame, 0.5)
        # print(frame.shape)
        if not ret:
            print ("Unable to capture video")
            return 
        kp_frame, des_frame = orb.detectAndCompute(img2, None)
        matches = bf.match(des_model, des_frame)
        matches = sorted(matches, key=lambda x: x.distance)
        if (len(matches) > MIN_MATCHES):
            src_pts = np.float32([kp_model[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            h, w = img1.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, homography)
            # frame = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
            if homography is not None:
                try:
                    # obtain 3D projection matrix from homography matrix and camera parameters
                    projection = pf.projection_matrix(camera_parameters, homography) 
                    # print("projection", projection) 
                    # project cube or model
                    frame = pf.render(frame, obj, projection, model, False)
                except:
                    pass
            
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