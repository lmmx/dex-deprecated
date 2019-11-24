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
    plot_fft_spectrum,
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

def BG_img(img):
    bg = grade(brighten(boost_contrast(scale_img(img))))
    return bg


def calculate_sparsities(crop_from_top=0.8, view=False, verbosity=1):
    """
    Calculate sparsities, after chopping off the top by a ratio
    of {crop_from_top} (e.g. 0.6 deducts 60% of image height).
    """
    for i in range(0, len(images)):
        print(f"Calculating sparsity for image {i}")
        im = imread(img_dir / images[i])
        crop_im = im[round(im.shape[0] * crop_from_top) :, :, :]
        bg = BG_img(crop_im)
        sel, sparsities = scan_sparsity(bg, VISUALISE=view, verbosity=verbosity)
        print()


def example_fft_plot():
    """
    Give an example usage of plot_fft_spectrum
    """
    img = imread(img_dir / images[0])
    bg = BG_img(img)
    plot_fft_spectrum(bg[3600:3700,300:800])
    return


def example_scan_fft(crop_from_top=0.8, view=False, verbosity=1):
    """
    Give an example usage of scan_sparsity to give another FFT plot.
    """
    im = imread(img_dir / images[0])
    crop_im = im[round(im.shape[0] * crop_from_top) :, :, :]
    bg = BG_img(crop_im)
    x_win = (300,800) # Default window for scan_sparsity if not provided
    sel, sparsities = scan_sparsity(bg, VISUALISE=view, verbosity=verbosity)
    sel_im = bg[sel[0][0]:sel[0][1], x_win[0]:x_win[1]]
    plot_fft_spectrum(sel_im)
    return


# Assemble into individual PDFs

# authors: [420-422].jpg
# topics: [423-431].jpg
# symbols: 432.jpg
