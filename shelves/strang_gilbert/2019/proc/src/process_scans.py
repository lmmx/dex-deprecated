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
from skimage.measure import find_contours

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

def get_contours(trimmed_img, view=False, level=254, connectedness="high", refine=True):
    """
    Pass in a trimmed gradient image (e.g. from `calculate_sparsity`) and get
    a list of contours found in that region, which can then be processed to
    find a single contour. This function is a user-friendly or 'front facing'
    version of contour_line() in `image_funcs.py`.
    """
    contours = find_contours(trimmed_img, level=level, fully_connected=connectedness)
    if refine:
        contours = refine_contours(contours)
    if view:
        plt.imshow(trimmed_img, cmap=plt.get_cmap('gray'))
        for n, contour in enumerate(contours):
            plt.plot(contour[:, 1], contour[:, 0], linewidth=2)
        plt.show()
    return contours

def plot_contours(trimmed_img, contours):
    plt.imshow(trimmed_img, cmap=plt.get_cmap('gray'))
    for n, contour in enumerate(contours):
        plt.plot(contour[:, 1], contour[:, 0], linewidth=2)
    plt.show()
    return

def refine_contours(contours, x_win=(300,800)):
    """
    Discard unnecessary contours from the contour set, by detecting overlap with the
    widest contour. Note that this could all be done on rounded coords but leave them
    as found since they are adjusted from int values for display purposes.
    """
    con_w = lambda c: np.max(c[:,1] - np.min(c[:,1]))
    contours_by_size = [c for c in reversed(sorted(contours, key=lambda x: len(x)))]
    contours_by_width = [c for c in reversed(sorted(contours, key=lambda x: con_w(x)))]
    contour_widths = [con_w(c) for c in contours_by_width]
    print(f"Contours by width: {contour_widths}")
    print(f"Contours by size: {[len(c) for c in contours_by_size]}")
    # The width is the difference in start and end point x values: coords are (y,x)
    max_width = max(x_win) - min(x_win) - 1
    biggest_c = contours_by_width[0]
    biggest_c_r = np.round(biggest_c)
    biggest_c_w = abs(biggest_c[-1][1] - biggest_c[0][1])
    big_c = contours_by_width[1]
    big_c_r = np.round(big_c)
    big_c_w = abs(big_c[-1][1] - big_c[0][1])
    contour_diff = np.where([val not in biggest_c_r for val in big_c_r])[0]
    if con_w(biggest_c_r) == max_width:
        print("Biggest one spans the window")
        return [biggest_c]
    else:
        # Consider combining the top 2 contours into one (may need coord deduplication)
        both_ways = [ [big_c, biggest_c], [biggest_c, big_c] ]
        bridgeable = np.any([np.max(a[:,1] >= np.min(b[:,1])) for (a,b) in both_ways])
        # Only makes sense to combine two whose ranges overlap (bridgeable by isthmus)
        if con_w(biggest_c_r) + con_w(big_c_r) >= max_width - 1:
            print("Biggest two span the window")
            if bridgeable:
                print("The two are bridgeable")
                if con_w(biggest_c_r) + con_w(big_c_r) == max_width - 1:
                    print("The two are adjacent, not overlapping (bridge by isthmus)")
                else:
                    print("The two are overlapping (bridge by merging)")
                return [biggest_c, big_c]
            else:
                print("These two can't be bridged though")
        else:
            print(f"Nah: biggest_c is {con_w(biggest_c)}, max_width is {max_width}")
    print(f"Non-overlap between the top 2 contours by size is {contour_diff.size}")
    return contours

def calculate_sparsity(img_n, crop_from_top=0.8, view=False, verbosity=1, x_win=(300,800)):
    """
    Calculate sparsities, after chopping off the top by a ratio
    of {crop_from_top} (e.g. 0.6 deducts 60% of image height).
    """
    print(f"Calculating sparsity for image {img_n}")
    im = imread(img_dir / images[img_n])
    crop_im = im[round(im.shape[0] * crop_from_top) :, :, :]
    bg = BG_img(crop_im)
    sel, sparsities = scan_sparsity(bg, x_win=x_win, VISUALISE=view, verbosity=verbosity)
    trimmed = bg[sel[0][0]:sel[0][1],x_win[0]:x_win[1]]
    return sel, sparsities, trimmed

def calculate_sparsities(crop_from_top=0.8, view=False, verbosity=1, x_win=(300,800)):
    """
    Calculate sparsities, after chopping off the top by a ratio
    of {crop_from_top} (e.g. 0.6 deducts 60% of image height).
    """
    for i in range(0, len(images)):
        print(f"Calculating sparsity for image {i}")
        im = imread(img_dir / images[i])
        crop_im = im[round(im.shape[0] * crop_from_top) :, :, :]
        bg = BG_img(crop_im)
        sel, sparsities = scan_sparsity(bg, x_win=x_win, VISUALISE=view, verbosity=verbosity)
        trimmed = bg[sel[0][0]:sel[0][1],x_win[0]:x_win[1]]
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

for img_n in range(0, len(images)):
    sel, sparsities, trimmed = calculate_sparsity(img_n, view=True)
    if None not in sel[0]:
        contours = get_contours(trimmed, view=True)
# For images[0] (0th index i.e. 1st image in the list),
# the long contour is the 6th index (i.e. 7th in the list), 528 points long
# The 2nd longest is 522 points long (very close, indistinguishable on len alone)

# Assemble into individual PDFs

# authors: [420-422].jpg
# topics: [423-431].jpg
# symbols: 432.jpg
