import cv2
import numpy as np

from ar_model import ARModel
import config
import process_func as pf


def project_3d_model_to_target_plane(ref, target):
    target.set_homography(ref)

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

    return frame


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

        cv2.imshow('After process', target.target_after)

        # cv2.drawKeypoints(frame_read, target.get_keypoints(),
        #   target.target, color=(0, 255, 0))

        if len(target.get_matches()) > config.MIN_MATCHES:
            frame_read = project_3d_model_to_target_plane(
                config.joker, target)

            # frame_matches = cv2.drawMatches(config.joker.ref_plane, config.joker.get_keypoints(), frame_read,
            #                                 target.get_keypoints(), target.get_matches()[:10], 0, flags=2)
            # cv2.imshow('After matches', frame_matches)

        else:
            print('Not enough matches found - {}/{}'.format(
                len(target.get_matches()), config.MIN_MATCHES))

        cv2.imshow('Frame', frame_read)
        if cv2.waitKey(50) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
