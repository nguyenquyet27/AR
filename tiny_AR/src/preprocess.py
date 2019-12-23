import cv2
import numpy as np


# Process input image to match the original line drawing
def image_proc(img, scale_factor):

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    # Luminance channel of HSV image
    lum = img_hsv[:, :, 2]

    # Adaptive thresholding
    lum_thresh = cv2.adaptiveThreshold(
        lum, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 15)

    # Remove all small connected components
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(
        lum_thresh, connectivity=8)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1
    min_size = 90*scale_factor

    lum_clean = np.zeros((output.shape))
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            lum_clean[output == i + 1] = 255

    # use mask to remove all neat outline of original image
    lum_seg = np.copy(lum)
    lum_seg[lum_clean != 0] = 0
    lum_seg[lum_clean == 0] = 255

    # Gaussian smoothing of the lines
    # lum_seg = cv2.GaussianBlur(lum_seg,(3,3),1)
    lum_seg = cv2.medianBlur(lum_seg, 3)
    return lum_seg
