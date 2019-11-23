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
assert doc_dir.exists() and doc_dir.is_dir()
assert img_dir.exists() and img_dir.is_dir()

images = [x.name for x in img_dir.iterdir() if x.suffix == ".jpg"]
assert len(images) > 0

test_img = imread(img_dir / images[0])

def BG_img(img):
    bg = grade(brighten(boost_contrast(scale_img(img))))
    return bg

brightened_graded = BG_img(test_img)
# Assemble into individual PDFs

# authors: [420-422].jpg
# topics: [423-431].jpg
# symbols: 432.jpg
