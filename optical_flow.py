import cv2
import time
import numpy as np

from ar_model import ARModel
import config
import process_func as pf


dist_coefs = np.array([-1.96413312e-01, -5.38486169e-01,
                       8.63538086e-03, -3.63210304e-03, 6.22198572e+00])


lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


def project_3d_model_to_target_plane(ref, target):
    dst_points = target.set_homography(ref)

    points = np.float32(
        [[0, 0],
         [0, ref.height - 1],
         [ref.width - 1, ref.height - 1],
         [ref.width - 1, 0]]
    ).reshape(-1, 1, 2)

    dst = cv2.perspectiveTransform(points, target.get_homography())

    frame = cv2.polylines(
        target.target, [np.int32(dst)], True, (255, 255, 255), 3, cv2.LINE_AA)
    if target.get_homography() is not None:
        try:
            # obtain 3D projection matrix from homography matrix and camera parameters
            projection = pf.projection_matrix(
                config.camera_intrinsic, target.get_homography())
            # project cube or model
            frame = pf.render(frame, config._3d_fox,
                              projection, ref.image_ref, False)
        except:
            pass

    return frame, dst_points, dst


if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-vb', default=str(0),
                    help="lookup cv2.VideoCapture for video backend parameters")
    args = ap.parse_args()

    cap = cv2.VideoCapture(int(args.vb))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.image_plane_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.image_plane_height)

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream.")

    while True:
        ret, frame_read = cap.read()

        target = ARModel(config.joker, frame_read)

        if target.get_descriptors() is None:
            cv2.imshow('Frame', frame_read)
            if cv2.waitKey(50) == 27:
                break
            continue

        target.set_matches(config.joker)

        cv2.imshow('After process', target.get_preprocess_target())

        # cv2.drawKeypoints(frame_read, target.get_keypoints(),
        #   target.target, color=(0, 255, 0))
        if len(target.get_matches()) > config.MIN_MATCHES:
            frame_read, dst_points, dst = project_3d_model_to_target_plane(
                config.joker, target)

            src_pts = np.copy(dst_points)
            pg_points = np.array([
                (dst[0][0][0], dst[0][0][1], 0.0),  # 1
                (dst[1][0][0], dst[1][0][1], 0.0),  # 2
                (dst[2][0][0], dst[2][0][1], 0.0),  # 3
                (dst[3][0][0], dst[3][0][1], 0.0)  # 4
            ])
            img2_old = np.copy(target.get_preprocess_target())
            break

            # frame_matches = cv2.drawMatches(config.joker.ref_plane, config.joker.get_keypoints(), frame_read,
            #                                 target.get_keypoints(), target.get_matches()[:10], 0, flags=2)
            # cv2.imshow('After matches', frame_matches)

        else:
            print('Not enough matches found - {}/{}'.format(
                len(target.get_matches()), config.MIN_MATCHES))

        cv2.imshow('Frame', frame_read)
        if cv2.waitKey(50) == 27:
            break

    while True:
        ret, frame_read = cap.read()

        target = ARModel(config.joker, frame_read)

        # Calculate optical flow
        dst_pts, st, err = cv2.calcOpticalFlowPyrLK(
            img2_old, target.get_preprocess_target(), src_pts, None, **lk_params)

        # Select good points
        good_new = dst_pts[st == 1]
        good_old = src_pts[st == 1]

        # Compute Homography
        H, mask = cv2.findHomography(good_old, good_new, cv2.RANSAC, 10.0)

        # Transform frame edge based on new homography
        dst = cv2.perspectiveTransform(dst, H)

        frame_read = cv2.polylines(
            target.target, [np.int32(dst)], True, (255, 255, 255), 3, cv2.LINE_AA)

        # Copy feature points and frame for processing of next frame
        src_pts = np.copy(good_new).reshape(-1, 1, 2)
        img2_old = np.copy(target.get_preprocess_target())

        # # Estimate the camera pose from frame corner points in world coordinates and image frame
        # # THe rotation vectors and translation vectors are obtained
        # ret, rvecs, tvecs, inlier_pt = cv2.solvePnPRansac(
        #     pg_points, dst, config.camera_intrinsic, dist_coefs)

        if H is not None:
            try:
                # obtain 3D projection matrix from homography matrix and camera parameters
                projection = pf.projection_matrix(
                    config.camera_intrinsic, H)
                # project cube or model
                frame_read = pf.render(frame_read, config._3d_fox,
                                       projection, config.joker.image_ref, False)
            except:
                pass

        cv2.imshow('Frame', frame_read)
        if cv2.waitKey(50) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
