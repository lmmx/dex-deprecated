import numpy as np
import matplotlib.pyplot as plt
from imageio import imread
from pathlib import Path
from math import sqrt
from numpy.linalg import norm
from image_funcs import (
    calculate_contrast,
    scale_img,
    boost_contrast,
    brighten,
    show_img,
    plot_img_hist,
    grade,
    show_bbox_overlay,
    contour_line,
    scan_sparsity,
)
from gradients import get_grads, get_grad, show_grad
from skimage.color import rgb2grey

doc_dir = Path("__file__").resolve().parent / "doc"
img_dir = Path("__file__").resolve().parent / "img"
dir_msg = "Please run this script from its parent directory, proc/"
assert doc_dir.exists() and doc_dir.is_dir(), dir_msg
assert img_dir.exists() and img_dir.is_dir(), dir_msg

images = [x.name for x in img_dir.iterdir() if x.suffix == ".jpg"]
assert len(images) > 0

test_img = imread(img_dir / images[0])


def BG_img(img):
    bg = grade(brighten(boost_contrast(scale_img(img))))
    return bg


brightened_graded = BG_img(test_img)
# Assemble into individual PDFs


def calculate_sparsities(crop_from_top=0.8, VISUALISE=False):
    """
    Calculate sparsities, after chopping off the top by a ratio
    of {crop_from_top} (e.g. 0.6 deducts 60% of image height).
    """
    for i in range(0, len(images)):
        print(f"Calculating sparsity for image {i}")
        im = imread(img_dir / images[i])
        crop_im = im[round(im.shape[0] * crop_from_top) :, :, :]
        bg = BG_img(crop_im)
        s = scan_sparsity(bg, VISUALISE=VISUALISE)
        print()


# authors: [420-422].jpg
# topics: [423-431].jpg
# symbols: 432.jpg
