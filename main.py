import cv2
import time
import argparse
import numpy as np

from ar_model import ARModel
import config
import process_func as pf

kt = 0
projection = None
homograp = np.ones((3,3))
def project_3d_model_to_target_plane(ref, target):
    global kt, projection, homograp
    target.set_homography(ref)
    if kt==0:
        homograp = target.get_homography()
    else:
        homograp = (target.get_homography()+homograp)/2
        # homograp = target.get_homography()
        kt = 1

    points = np.float32(
        [[0, 0],
         [0, ref.height - 1],
         [ref.width - 1, ref.height - 1],
         [ref.width - 1, 0]]
    ).reshape(-1, 1, 2)

    dst = cv2.perspectiveTransform(points, homograp)
    
    frame = cv2.polylines(
        target.target, [np.int32(dst)], True, (255, 255, 255), 3, cv2.LINE_AA)
    # frame = target.target
    if homograp is not None:
        try:
            # obtain 3D projection matrix from homography matrix and camera parameters
            projection = pf.projection_matrix(
                config.camera_intrinsic, homograp)
            # project cube or model
            frame = pf.render(frame, config._3d_fox,
                              projection, ref.image_ref, False)
        except:
            pass

    return frame


if __name__ == "__main__":
    

    # Parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-vb', default=str(0),
                    help="lookup cv2.VideoCapture for video backend parameters")
    args = ap.parse_args()
    start_time = time.time()
    cap = cv2.VideoCapture(int(args.vb))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.image_plane_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.image_plane_height)

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream.")

    while True:
        ret, frame_read = cap.read()

        target = ARModel(config.joker, frame_read)
        target2 = ARModel(config.joker2, frame_read)
        if target.get_descriptors() is None:
            cv2.imshow('Frame', frame_read)
            if cv2.waitKey(50) == 27:
                break
            continue
        frame_read2 = frame_read
        target.set_matches(config.joker)
        target2.set_matches(config.joker2)

        cv2.imshow('After process', target.target_after)

        # cv2.drawKeypoints(frame_read, target.get_keypoints(),
        #   target.target, color=(0, 255, 0))       
        if len(target.get_matches()) > config.MIN_MATCHES:
            frame_read = project_3d_model_to_target_plane(
                config.joker, target)
        if len(target2.get_matches()) > config.MIN_MATCHES:
            frame_read2 = project_3d_model_to_target_plane(
                config.joker2, target2)
            # frame_matches = cv2.drawMatches(config.joker.ref_plane, config.joker.get_keypoints(), frame_read,
            #                                 target.get_keypoints(), target.get_matches()[:10], 0, flags=2)
            # cv2.imshow('After matches', frame_matches)
        cv2.addWeighted(frame_read,0.5,frame_read2,0.5,0)
        print('Not enough matches found - {}/{}'.format(
        len(target.get_matches()), config.MIN_MATCHES))

        cv2.imshow('Frame', frame_read)
        end_time = time.time()
        print ('total run-time: %f ms' % ((end_time - start_time) * 1000))
        if cv2.waitKey(50) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
