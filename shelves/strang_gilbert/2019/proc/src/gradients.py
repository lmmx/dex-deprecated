# Note: adapted from src/waterdown.py in GitHub repo lmmx/waterdown

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from imageio import imread
from skimage.color import rgb2grey
from cv2 import Canny, imread as cv_imread
from scipy.ndimage import sobel
from skimage.measure import find_contours

###################### Basic image functions ##########################

def show_image(img, bw=False, alpha=1, no_ticks=True, title='', suppress_show=True):
    """
    Show an image using a provided pixel row array.
    If bw is True, displays single channel images in black and white.
    """
    if not bw:
        plt.imshow(img, alpha=alpha)
    else:
        plt.imshow(img, alpha=alpha, cmap=plt.get_cmap('gray'))
    if no_ticks:
        plt.xticks([]), plt.yticks([])
    if title != '':
        plt.title = title
    if not suppress_show:
        plt.show()
    return

def save_image(image, figsize, save_path, ticks=False, grey=True):
    """
    Save a given image in a given location, default without ticks
    along the x and y axis, and if there's only one channel
    (i.e. if the image is greyscale) then use the gray cmap
    (rather than matplot's default Viridis).
    """
    fig = plt.figure(figsize=figsize)
    if grey:
        plt.imshow(image, cmap=plt.get_cmap('gray'))
    else:
        plt.imshow(image)
    if not ticks:
        plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    fig.savefig(save_path)
    return

################# Image gradients and edge detection #############

def get_grads(img):
    """
    Convolve Sobel operator independently in x and y directions,
    to give the image gradient.
    """
    dx = sobel(img, 0)  # horizontal derivative
    dy = sobel(img, 1)  # vertical derivative
    return dx, dy

def get_grad(img, normalise_rgb=False):
    if len(img.shape) == 3 and img.shape[-1] == 3:
        # Must convert RGB images to single channel
        img = rgb2grey(img)
    dx, dy = get_grads(img)
    mag = np.hypot(dx, dy)  # magnitude
    if normalise_rgb:
        mag *= 255.0 / np.max(mag)
    return np.uint8(mag)

def show_grad(img):
    grad = get_grad(img)
    plt.imshow(grad, cmap=plt.get_cmap('gray'))
    plt.show()
    return

def auto_canny(image, sigma=0.4):
    """
    Zero parameter automatic Canny edge detection courtesy of
    https://www.pyimagesearch.com - use a specified sigma value
    (taken as 0.4 from Dekel et al. at Google Research, CVPR 2017)
    to compute upper and lower bounds for the Canny algorithm
    along with the median of the image, returning the edges.
    
    See the post at the following URL:
    https://www.pyimagesearch.com/2015/04/06/zero-parameter-
    automatic-canny-edge-detection-with-python-and-opencv/
    """
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = Canny(image, lower, upper)
    return edged

def bbox(img):
    """
    Return a bounding box (rmin, rmax, cmin, cmax). To retrieve the
    bounded region, access `image[rmin:rmax+1, cmin:cmax+1]`.
    """
    a = np.where(img != 0)
    bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
    return bbox

def contour_line(img, level=254, connectedness="high"):
    """
    Contour a binary image, used to turn the line into a single
    connected entity. Connectedness "high" gives 8-connectedness,
    and "low" gives 4-connectedness.
    """
    contours = find_contours(img, level=level, fully_connected=connectedness)
    plt.imshow(img, cmap=plt.get_cmap('gray'))
    for n, contour in enumerate(contours):
        plt.plot(contour[:, 1], contour[:, 0], linewidth=2)
    plt.show()
    return

def estimate_contour_sparsity(y_win_size, x_win_size, c_len=0.9, c_width=1):
    """
    Given the size of a scanline window used in scan_sparsity(),
    estimate the expected maximum sparsity of the window containing
    the contour line for a page, so as to distinguish between empty
    regions and sparse ones.

    Default: estimate for a line width of 1 pixel, and for a line expected to
    stretch across 90% of the total width of the window (if expecting it to
    span the full width [when contoured with skimage.measure.find_contours],
    e.g. if using a span away from the sides of the image, use c_len=1.0).
    """
    estim = c_len * c_width / (y_win_size * x_win_size)
    sparsity = 1 - estim
    return sparsity

def scan_sparsity(img, win_size=100, x_win=None, verbose=True, estim_c_len=1.):
    """
    Calculate and print sparsity values for windows of scanlines
    (default is 100 scanlines at a time). Additionally, find 'unconnected columns'
    and quantify how wide they are (i.e. the maximum number of unconnected columns
    in each window of scanlines). These unconnected columns represent the
    non-connectedness of the scanline window across this region, and are grounds for
    excluding the region from the search for the page contour.
    """
    if x_win is None:
        x_win = (300, 800)
    if verbose:
        print(f"Using x_win values of {x_win}, y window sizes of {win_size}")
    x_win_start, x_win_end = x_win
    cutoff = estimate_contour_sparsity(win_size, abs(np.subtract(*x_win)), estim_c_len)
    sparsities = []
    for win in np.arange(0, len(img), win_size):
        im = img[win:win+win_size, x_win_start:x_win_end]
        sparsity = np.sum(im == 0) / np.prod(im.shape[0:2])
        if verbose:
            if sparsity < cutoff:
                print(f"Rows {win}-{win+win_size}: sparsity = {100 * sparsity}%")
            else:
                print(f"Rows {win}-{win+win_size} rejected (too sparse: {sparsity})")
        sparsities.append([(win,win+win_size),sparsity])
    return sparsities

###################### Image channel functions #########################

def to_rgb(im):
    """
    Turn a single valued array to a 3-tuple valued array of RGB, via
    http://www.socouldanyone.com/2013/03/converting-grayscale-to-rgb
    -with-numpy.html (solution 1a)
    """
    w, h = im.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 2] = ret[:, :, 1] = ret[:, :, 0] = im
    return ret
