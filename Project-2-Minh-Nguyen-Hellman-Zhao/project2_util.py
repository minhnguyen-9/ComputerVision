import os
import sys
from collections import namedtuple
import json

import cv2
import numpy as np
import matplotlib.pyplot as plt

MAX_DISPLAY_W = 1200 
MAX_DISPLAY_H = 700

FIRST_IMSHOW = True

############################################################################################
# Image manipulation functions

######################################################################
def draw_image_with_mask(image, mask):
    """Return a copy of image with the mask overlaid for display."""
    assert image.shape[:2] == mask.shape
    return alpha_blend(image // 2, image // 2 + 128, mask)


######################################################################
def alpha_blend(img1, img2, mask):
    """Perform alpha blend of img1 and img2 using mask.

    Result is an image of same shape as img1 and img2.  Wherever mask
    is 0, result pixel is same as img1. Wherever mask is 255 (or 1.0
    for float mask), result pixel is same as img2. For values in between,
    mask acts as a weight for a weighted average of img1 and img2.

    See https://en.wikipedia.org/wiki/Alpha_compositing
    """

    (h, w) = img1.shape[:2]

    assert img2.shape == img1.shape
    assert mask.shape == img1.shape or mask.shape == (h, w)

    result = np.empty_like(img1)

    if mask.dtype == np.uint8:
        mask = mask.astype(np.float32) / 255.0

    if len(mask.shape) == 2 and len(img1.shape) == 3:
        mask = mask[:, :, None]

    result[:] = img1 * (1 - mask) + img2 * mask

    return result


############################################################################################
# Functions for creating ROIs

######################################################################
def ellipse_mask_from_roi(src_image, src_roi, wh_scales=(1.0, 1.0), flip=False):
    src = src_roi
    wsz, hsz = wh_scales

    h, w = src_image.shape[:2]
    src_size = (w, h)

    ellipse_mask = roi_draw_ellipse(src, wsz, hsz, src_size)
    return ellipse_mask

######################################################################

def roi_from_points(top_left, top_right, bottom):
    """Create an ImageROI struct from three points given by user.
    Returns a namedtuple with fields: 

      * center:         center of ROI rectangle as (float, float) tuple
      * angle:          angle of ROI rectangle in radians
      * width:          width of ROI rectangle
      * height:         height of ROI rectangle, also used as 
                        scaling factor for warps
    """
    p0 = np.array(top_left, dtype=np.float32)
    p1 = np.array(top_right, dtype=np.float32)
    p2 = np.array(bottom)

    u = p1-p0
    width = np.linalg.norm(u)
    u /= width
    v = p2-p0

    if u[0] * v[1] - u[1] * v[0] < 0:
        u = -u
        top_left, top_right = top_right, top_left

    v -= u * np.dot(u, v) 

    assert np.abs(np.dot(u, v)) < 1e-4
    height = np.linalg.norm(v)

    cx, cy = p0 + 0.5*u*width + 0.5*v
    angle = np.arctan2(u[1], u[0])

    return ImageROI((float(cx), float(cy)), 
                     float(angle), float(width), float(height))



############################################################################################
# Region of interest handlers

######################################################################
ImageROI = namedtuple(
    'ImageROI', 
    ['center', 'angle', 'width', 'height']
) # Region of Interest container object


######################################################################
def roi_from_center_angle_dims(center, angle, width, height):
    """Simple ROI constructor from center, angle, width, height."""
    center = (float(center[0]), float(center[1]))
    angle = float(angle)
    width = float(width)
    height = float(height)

    return ImageROI(center, angle, width, height)


######################################################################
def roi_get_matrix(image_roi):
    """Get a 3x3 matrix mapping local object points (x, y) in the ROI to
    image points (u, v) according to the formulas:

       x' = image_roi.height * x
       y' = image_roi.height * y

       c  = cos(image_roi.angle)
       s  = sin(image_roi.angle)

       u  = c * x' - s * y' + image_roi.center[0]
       v  = s * x' + c * y' + image_roi.center[1]

    """
    c = np.cos(image_roi.angle)
    s = np.sin(image_roi.angle)
    tx, ty = image_roi.center
    h = image_roi.height
    return np.array([[c*h, -s*h, tx],
                     [s*h, c*h, ty],
                     [0, 0, 1]])


######################################################################

def roi_map_points(image_roi, opoints):
    """Map from local object points to image points using the matrix
    established by roi_get_matrix(). The opoints parameter should be an
    n-by-2 array of (x, y) object points. The return value is an
    n-by-2 array of (u, v) pixel locations in the image.

    """
    M = roi_get_matrix(image_roi)
    opoints = opoints.reshape(-1, 1, 2)
    ipoints = cv2.perspectiveTransform(opoints, M)
    return ipoints.reshape(-1, 2)

######################################################################
def draw_roi_on_image(image, image_roi, color=(255, 255, 0), thickness=10):
    """Draws ROI box on image, accounting for angle. Takes in optional color and thickness."""
    opoints = np.array([
        [-0.5, -0.5],
        [ 0.5, -0.5],
        [ 0.5,  0.5],
        [-0.5,  0.5],
        [-0.2,  0.0],
        [ 0.2,  0.0],
        [ 0.0, -0.2],
        [ 0.0,  0.2],
        [ 0.0,  0.5]
    ]) * np.array([image_roi.width/image_roi.height, 1])

    ipoints = roi_map_points(image_roi, opoints).astype(int)

    display = image.copy()
    scl = thickness

    cv2.polylines(display, [ipoints[:4]], True, 
                  color, scl, cv2.LINE_AA)


    for i in [0, 1, -1]:
        cv2.circle(display, tuple(ipoints[i]), 4*scl, 
                   color, scl, cv2.LINE_AA)

    cv2.line(display, tuple(ipoints[4]), tuple(ipoints[5]), 
             color, scl, cv2.LINE_AA)

    cv2.line(display, tuple(ipoints[6]), tuple(ipoints[7]), 
             color, scl, cv2.LINE_AA)

    return display


######################################################################
def roi_draw_ellipse(img_roi, wsz, hsz, size=None): 
    """Draw an ellipse into an 8-bit single-channel mask image centered
    on the given ROI and rotated to align with it. The given dimensions
    are as fractions of the total height of the original ROI.
    """
    w, h = size

    mask = np.zeros((h, w), dtype=np.uint8)
    
    axes = 0.5 * img_roi.height * np.array([wsz, hsz])

    center = tuple([int(x) for x in img_roi.center])
    axes = tuple([int(x) for x in axes])

    deg = 180/np.pi

    return cv2.ellipse(mask, center, axes,
                       img_roi.angle*deg, 0, 360,
                       (255, 255, 255), -1, cv2.LINE_AA)


######################################################################
def roi_warp(src_image, src_roi, dst_roi, dst_size=None, flip=False):

    """Warps the src_image so that its ROI overlaps the corresponding ROI
    in the destination image. Image scaling is based on height.
    """

    if src_image is None:
        src_image = cv2.imread(src_roi.image_filename)

    if dst_size is None:
        dst_image = cv2.imread(dst_roi.image_filename)
        h, w = dst_image.shape[:2]
        dst_size = (w, h)

    src_mat = roi_get_matrix(src_roi)
    dst_mat = roi_get_matrix(dst_roi)

    if flip:
        flip = np.diag([-1, 1, 1])
    else:
        flip = np.eye(3)

    M = dst_mat @ flip @ np.linalg.inv(src_mat) 

    return cv2.warpAffine(src_image, M[:2], dst_size, 
                          flags=cv2.INTER_LINEAR, 
                          borderMode=cv2.BORDER_REFLECT_101)


############################################################################################
# Crop/Warp functions for ROIs

######################################################################
def crop_function(src_image, src_roi, result_height, wh_scales=(1.0, 1.0), scroll_xy=(0, 0), flip=False):
    """Crop function -- takes in image and region of interest
    Inputs:
        src_image: np.array of the image to crop (grayscale or color)
        src_roi: ImageROI object specifying part of image to crop
        result_height: integer, specifying output height
        wh_scales: 2-tuple of floats, specifying how much to expand ROI in w, h direction (respectively)
        scroll_xy: 2-tuple of integers, specifying how much to offset the ROI in x, y direction (respectively)
        flip: boolean, specifying weather or not to flip the image in the x direction
    Returns:
        result_image: Resulting cropped image
        dst_roi: ROI that was modified by wh_scales, scroll_xy
    """

    wsz, hsz = wh_scales
    scroll_x, scroll_y = scroll_xy
    
    wpx = wsz * src_roi.height
    hpx = hsz * src_roi.height

    result_width = int(round(result_height * wpx / hpx))

    scl = result_height / (src_roi.height * hsz)

    dst_roi = roi_from_center_angle_dims((0.5*result_width + scroll_x, 0.5*result_height + scroll_y),
                                         0.0,
                                         src_roi.width * scl,
                                         src_roi.height * scl)

    dst_size = (result_width, result_height)

    result_image = roi_warp(src_image, src_roi, dst_roi, dst_size, flip)
    return result_image, dst_roi


######################################################################
def warp_helper_function(src_img, dst_img, src_roi, dst_roi, flip=False):
    """Warp function -- maps ROI from one image onto another.
    Inputs:
        src_img: np.array of the image to warp ROI from
        dst_img: np.array of image to warp ROI to
        src_roi: ImageROI of warp ROI from
        dst_roi: ImageROI of warp ROI to
        flip: boolean, specifying weather or not to flip the image in the x direction
    """
    h, w = dst_img.shape[:2]
    warped = roi_warp(src_img, src_roi, dst_roi, dst_size=(w, h), flip=flip)
    dst_roi = roi_from_center_angle_dims(dst_roi.center,
                                         dst_roi.angle,
                                         src_roi.width * dst_roi.height / src_roi.height,
                                         dst_roi.height)
    return warped, dst_roi



######################################################################
def overlay_display(image1, image2, n_images=6):
    if np.any(image1.shape != image2.shape):
        raise ValueError("image1 and image2 need to be same size but got: {} and {}".format(image1.shape, image2.shape))
    if n_images < 2:
        raise ValueError("n_images needs to be >= 2 but got: {}".format(n_images))
    blend = np.empty_like(image1)
    t = 0.0

    fig, ax = plt.subplots(1, n_images, figsize=(5*n_images, 5))
    fig.suptitle('Overlay', fontsize=14)
    for i in range(n_images):
        # t = (np.pi*i)/(n_images-1)
        # u = np.cos(t) * 0.5 + 0.5 # Alternative interpolation
        u = (1.0*i)/(n_images-1)
        blend[:] = image1*(1.0 - u) + image2*u
        if len(blend.shape) == 2 or (len(blend.shape) == 3 and blend.shape[-1] == 1):
            ax[i].imshow(blend, cmap='gray')
        else:
            ax[i].imshow(blend)



############################################################################################
# Laplacian Visualization

######################################################################
def visualize_pyramid(lp, padding=8):
    """Utility function to display a Laplacian pyramid."""

    n = len(lp)-1
    outputs = []

    h, w = lp[0].shape[:2]

    hmax = max([li.shape[0] for li in lp])

    hstackme = []

    hpadding = np.full((hmax, padding, 3), 255, np.uint8)

    for i, li in enumerate(lp):

        assert li.dtype == np.float32

        if i == n:
            display = li
        else:
            display = 127 + li

        display = np.clip(display, 0, 255).astype(np.uint8)

        h, w = display.shape[:2]

        if h < hmax:
            vpadding = np.full((hmax - h, w, 3), 255, np.uint8)
            display = np.vstack((display, vpadding))


        if i > 0:
            hstackme.append(hpadding)

        hstackme.append(display)

    return np.hstack(tuple(hstackme))



