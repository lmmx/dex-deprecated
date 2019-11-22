import numpy as np
import matplotlib.pyplot as plt
from imageio import imread
from pathlib import Path
from math import sqrt
from numpy.linalg import norm
from skimage.filters import threshold_yen
from skimage.exposure import rescale_intensity
from skimage.color import rgb2grey
from gradients import auto_canny, get_grad, to_rgb, show_image, bbox


def calculate_contrast(img):
    # Contrast = sqrt(sum(I - I_bar)^2 / (M*N) )
    # See https://en.wikipedia.org/wiki/Contrast_(vision)#RMS_contrast
    if np.max(img) > 1:
        img = np.divide(img, [255, 255, 255])
    c = norm(img - np.mean(img, axis=(0, 1))) / sqrt(img.shape[0] * img.shape[1])
    return c


def scale_img(img, verbose=False):
    """
    Increase the contrast of an image by scaling its min and max.
    Returns a value between 0 and 1.
    """
    i_min = np.min(img, axis=(0, 1))
    i_max = np.max(img, axis=(0, 1))
    scaled = (img - i_min) / (i_max - i_min)
    return scaled


def brighten(img, yen_thresholding=True, unit_range=True):
    """
    After contrast has been increased, it can be boosted further by thresholding.
    If `yen_thresholding` is False, 2nd and 98th percentiles are used instead.
    """
    if yen_thresholding:
        yen_threshold = threshold_yen(img)
        in_range = (0, yen_threshold)
    else:
        in_range = np.percentile(img, (2,98))
    if unit_range:
        min_max = (0, 1)
    else:
        min_max = (0, 255)
    brightened = rescale_intensity(img, in_range, min_max)
    return scale_img(brightened)


def boost_contrast(img):
    """
    After contrast has been increased, it can be boosted further by thresholding.
    """
    boosted = img * (img - np.mean(img))
    return scale_img(boosted)


def grade(img, make_grey=True, make_uint8=True, sigma=None):
    """
    Set up an image for gradient calculation.
    Assumes image is either scaled at 0-1 or 0-255, and will
    convert to uint8 by scaling up to 255 if necessary.
    """
    if sigma is None: sigma = 0.4
    if make_grey: img = rgb2grey(img)
    if make_uint8 and img.dtype != 'uint8':
        if np.max(img) == 1.: img *= 255
        img = np.uint8(img)
    graded = auto_canny(img, sigma=sigma)
    return graded



def show_bbox_overlay(img, canny=True):
    """
    Estimate a bounding box for the page crop and show it.
    Input image should be a median 'edged' image, not original photo.
    If 'canny' parameter is True, use original auto-Canny method,
    otherwise calculate gradient.
    """
    input_img = np.copy(img)
    img = 255 * brighten(img)
    if canny:
        edged = grade(img)
    else:
        edged = get_grad(rgb2grey(img))
    bb = bbox(edged)
    green = to_rgb(rgb2grey(np.copy(img)))
    green[bb[0]:bb[1]+1, bb[2]:bb[3]+1] = [0, 100, 0]
    show_image(input_img, alpha=1)
    #plt.imshow(edged, cmap=plt.get_cmap('gray'), alpha=0.1)
    plt.imshow(green, alpha=0.4)
    plt.show()
    return


def show_img(img, bw=False, no_ticks=False, title=None):
    if not bw:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap=plt.get_cmap('gray'))
    if no_ticks: plt.xticks([]), plt.yticks([])
    if title is not None: plt.title = title
    plt.show()
    return


def plot_img_hist(img):
    for i in range(0, img.shape[-1]):
        plt.hist(img[:, :, i])
    plt.show()
    return
