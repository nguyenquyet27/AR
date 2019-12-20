import cv2
import math
import numpy as np

# Process input image to match the original line drawing
def image_proc(img,scale_factor):

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    #Luminance channel of HSV image
    lum = img_hsv[:,:,2]

    #Adaptive thresholding
    lum_thresh = cv2.adaptiveThreshold(lum,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,11,15)

    #Remove all small connected components
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(lum_thresh, connectivity=8)
    sizes = stats[1:, -1]; nb_components = nb_components - 1
    min_size = 90*scale_factor

    lum_clean = np.zeros((output.shape))
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            lum_clean[output == i + 1] = 255

    # use mask to remove all neat outline of original image
    lum_seg = np.copy(lum)
    lum_seg[lum_clean!=0] = 0
    lum_seg[lum_clean==0] = 255

    # Gaussian smoothing of the lines
    lum_seg = cv2.GaussianBlur(lum_seg,(3,3),1)
    lum_seg = cv2.medianBlur(lum_seg,3)
    return lum_seg

# Compute the homography
def computeHomography(src_pts, dst_pts):
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    return M

# Draw lines for frame boundary
def draw_frame(img,dst):
    img = cv2.polylines(img, [np.int32(dst)], True, (255, 0, 0), 3, cv2.LINE_AA)
    return img

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
    # print("col_2",col_2)
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

def render(img, obj, projection, model, color=False):
    """
    Render a loaded obj model into the current video frame
    """
    vertices = obj.vertices
    scale_matrix = np.eye(3) * 3
    h, w = model.shape

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        # render model in the middle of the reference surface. To do so,
        # model points must be displaced
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)
        if color is False:
            cv2.fillConvexPoly(img, imgpts, (137, 27, 211))
        else:
            color = hex_to_rgb(face[-1])
            color = color[::-1]  # reverse
            cv2.fillConvexPoly(img, imgpts, color)

    return img

def hex_to_rgb(hex_color):
    """
    Helper function to convert hex strings to RGB
    """
    hex_color = hex_color.lstrip('#')
    h_len = len(hex_color)
    return tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))
    
