import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from imageio import imread
from pathlib import Path
from math import sqrt
from numpy.linalg import norm
from skimage.filters import threshold_yen
from skimage.exposure import rescale_intensity
from skimage.color import rgb2grey
from gradients import auto_canny, get_grad, to_rgb, show_image, bbox
from skimage.measure import find_contours
from scipy.fftpack import fft2


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
        in_range = np.percentile(img, (2, 98))
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
    if sigma is None:
        sigma = 0.4
    if make_grey:
        img = rgb2grey(img)
    if make_uint8 and img.dtype != "uint8":
        if np.max(img) == 1.0:
            img *= 255
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
    green[bb[0] : bb[1] + 1, bb[2] : bb[3] + 1] = [0, 100, 0]
    show_image(input_img, alpha=1)
    # plt.imshow(edged, cmap=plt.get_cmap('gray'), alpha=0.1)
    plt.imshow(green, alpha=0.4)
    plt.show()
    return


def show_img(img, bw=False, no_ticks=False, title=None):
    if not bw:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap=plt.get_cmap("gray"))
    if no_ticks:
        plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title = title
    plt.show()
    return


def plot_img_hist(img):
    for i in range(0, img.shape[-1]):
        plt.hist(img[:, :, i])
    plt.show()
    return


def contour_line(img, level=254, connectedness="high"):
    """
    Contour a binary image, used to turn the line into a single
    connected entity. Connectedness "high" gives 8-connectedness,
    and "low" gives 4-connectedness.
    """
    contours = find_contours(img, level=level, fully_connected=connectedness)
    plt.imshow(img, cmap=plt.get_cmap("gray"))
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


def trim_window(win, verbose=False):
    """
    Given a window (a gradient image) chosen to contain the line of interest,
    trim off any blank rows from the top or the bottom of the window, to focus
    on the feature of interest.
    """
    assert 0 not in win.shape
    if np.max(win[0:1, :], axis=1) == 0:
        # There is at least 1 blank line at the top
        n_from_top = np.where(np.max(win, axis=1) != 0)[0][0]
    else:
        n_from_top = 0
    if np.max(win[-2:-1, :], axis=1) == 0:
        # There is at least 1 blank line at the bottom
        n_from_bottom = np.where(np.max(win[::-1, :], axis=1) != 0)[0][0]
    else:
        n_from_bottom = 0
    if n_from_bottom == 0:
        trimmed = win[0 + n_from_top :, :]
    else:
        trimmed = win[0 + n_from_top : 0 - n_from_bottom, :]
    if verbose:
        print(f"Trimmed window by {n_from_top} from top, {n_from_bottom} from bottom")
    assert 0 not in trimmed.shape
    return trimmed, n_from_top, n_from_bottom


def scan_sparsity(
    img, win_size=100, x_win=None, verbosity=1, estim_c_len=1.0, VISUALISE=False
):
    """
    Calculate and print sparsity values for windows of scanlines
    (default is 100 scanlines at a time). Additionally, find 'unconnected columns'
    and quantify how wide they are (i.e. the maximum number of unconnected columns
    in each window of scanlines). These unconnected columns represent the
    non-connectedness of the scanline window across this region, and are grounds for
    excluding the region from the search for the page contour.
    Verbosity can be 0, 1, or 2 for differing levels of reported output.
    """
    if verbosity == 1:
        verbose = True
        v_verbose = False
    elif verbosity == 2:
        verbose = v_verbose = True
    else:
        verbose = v_verbose = False
    if x_win is None:
        x_win = (300, 800)
    if v_verbose:
        print(f"Using x_win values of {x_win}, y window sizes of {win_size}")
    x_win_start, x_win_end = x_win
    cutoff = estimate_contour_sparsity(win_size, abs(np.subtract(*x_win)), estim_c_len)
    if v_verbose:
        print(f"Using cutoff value of {100*cutoff}%")
    sparsities = []
    selections = []
    clip_ratio = 0.6
    for win in np.arange(0, len(img), win_size // 2):
        im = img[win : win + win_size, x_win_start:x_win_end]
        sparsity = np.sum(im == 0) / np.prod(im.shape[0:2])
        full_span = np.all(np.max(im, axis=0) != 0)
        if not full_span:
            midsparse = 0
            sideconn = 0
            clip_rise = 0
            if v_verbose:
                print(f"Rows {win}-{win+win_size} rejected (too sparse: {sparsity})")
        elif sparsity < cutoff:
            fft_maxima = prune_fft(im)
            midsparse = calculate_mid_sparsity(fft_maxima)
            if midsparse > 0.01:
                sideconn = calculate_side_connection(fft_maxima)
                clipped_sideconn = calculate_side_connection(fft_maxima, clip_ratio)
                if 0 in (sideconn, clipped_sideconn):
                    clip_rise = 0
                else:
                    clip_rise = 100 * (clipped_sideconn - sideconn) / sideconn
            else:
                sideconn = 0
                clipped_sideconn = 0
                clip_rise = 0
            if v_verbose:
                print(
                    f"Rows {win}-{win+win_size}: sparsity = {100*sparsity:.2f}%; "
                    + f"FFT mid-sparse: {100*midsparse:.2f}%; "
                    + f"FFT side conn.: {100*sideconn:.2f}%"
                    + f"(clipped at {clip_ratio*100:.0f}%): "
                    + f"{100*clipped_sideconn:.2f}% ({clip_rise:.0f}% change)"
                )
        else:
            midsparse = 0
            sideconn = 0
            clip_rise = 0
            if v_verbose:
                print(f"Rows {win}-{win+win_size} rejected (too sparse: {sparsity})")
        span_bit = int(full_span)
        sparsities.append(
            [(win, win + win_size), sparsity, midsparse, sideconn, clip_rise, span_bit]
        )
    most_midsparse_window = list(reversed(sorted(sparsities, key=lambda k: k[2])))[
        0
    ][0]
    bad_midspars = np.all([s[2] == 0 for s in sparsities])
    bad_sideconn = np.all([s[3] == 0 for s in sparsities])
    if bad_sideconn:
        if verbose:
            print(
                " Side connection not used to distinguish: beware trusting only midsparsity!"
            )
            if bad_midspars:
                print(" Uh oh, both measures failed! No prediction can be made.")
        most_sideconn_window = most_midsparse_window
    else:
        most_sideconn_window = list(
            reversed(sorted(sparsities, key=lambda k: k[3]))
        )[0][0]
    if (
        not (bad_midspars and bad_sideconn)
        and most_midsparse_window == most_sideconn_window
    ):
        chosen_start, chosen_end = most_midsparse_window
        pretrim = img[chosen_start:chosen_end, x_win_start:x_win_end]
        trimmed, start_offset, end_offset = trim_window(pretrim)
        chosen_start += start_offset
        chosen_end -= end_offset
        selections.append((chosen_start, chosen_end))
        if verbose:
            print(f"The predictions agree: {chosen_start}-{chosen_end}")
        if v_verbose: print(selections[-1])
        if VISUALISE:
            # show_img(trimmed)
            contour_line(trimmed)
    elif bad_midspars and bad_sideconn:
        selections.append((None, None))
        if verbose:
            print("Hmm, who knows about this one")
        if v_verbose: print(selections[-1])
        # for s in sparsities: print(s)
    else:
        mmw_start, mmw_end = most_midsparse_window
        if mmw_start in range(*most_sideconn_window) or mmw_end in range(
            *most_sideconn_window
        ):
            chosen_start = min(min(most_midsparse_window, most_sideconn_window))
            chosen_end = max(max(most_midsparse_window, most_sideconn_window))
            pretrim = img[chosen_start:chosen_end, x_win_start:x_win_end]
            trimmed, start_offset, end_offset = trim_window(pretrim)
            chosen_start += start_offset
            chosen_end -= end_offset
            selections.append((chosen_start, chosen_end))
            if verbose:
                print(
                    f"Merged adjacent windows: new window {chosen_start}-{chosen_end}"
                )
            if v_verbose: print(selections[-1])
            if VISUALISE:
                # show_img(trimmed)
                contour_line(trimmed)
        else:
            if verbose:
                print(
                    "Uh oh, the predictions disagree!"
                )
                print(
                    f"Based on FFT midsparsity, predicted window for the "
                    +f"line is {most_midsparse_window}"
                )
            mmw_midsparsity = [
                s for s in sparsities if s[0][0] == most_midsparse_window[0]
            ][0][2]
            if verbose:
                print(
                    f"--> The FFT midsparsity for "
                    +f"{most_midsparse_window} is {mmw_midsparsity}"
                )
                print(
                    f"Based on FFT sideconnection, predicted window "
                    +f"for the line is {most_sideconn_window}"
                )
            msw_sideconn = [
                s for s in sparsities if s[0][0] == most_sideconn_window[0]
            ][0][3]
            if verbose:
                print(
                    f"--> The FFT sideconnection for "
                    +f"{most_sideconn_window} is {msw_sideconn}"
                )
    return sparsities


def calculate_mid_sparsity(img_fft):
    col_sums = np.sum(img_fft, axis=0)
    mid_x = len(col_sums) // 2
    mid_sparse_x_start = reversed(col_sums)
    # Get the index of the first nonzero entry away from the x midpoint
    first_nonzero = np.where(col_sums[::-1][mid_x:] != 0)[0][0]
    zero_col_start = mid_x - first_nonzero
    # Get the index of the last nonzero entry away from the first zero entry
    zero_col_end = zero_col_start + np.where(col_sums[zero_col_start:] != 0)[0][0]
    sparsity = (zero_col_end - zero_col_start) / len(col_sums)
    return sparsity


def calculate_side_connection(pruned, clip_ratio=None):
    """
    Calculate the proportion of FFT side pixels that are maximal (i.e. in the
    top quintile obtained after pruning), which is being used to determine the
    presence of a line [page boundary].
    """
    left_side_end = np.where(np.max(pruned, axis=0) == 0)[0][0]
    if clip_ratio is not None:
        left_side_end = int(round(left_side_end * clip_ratio))
    r_offset = np.where(np.max(np.fliplr(pruned), axis=0) == 0)[0][0]
    if clip_ratio is not None:
        r_offset = int(round(r_offset * clip_ratio))
    right_side_start = pruned.shape[1] - r_offset
    # Max. values are in cols (0:left_side_end), (right_side_start:pruned.shape[1]
    l_side = pruned[:, 0:left_side_end]
    r_side = pruned[:, right_side_start : pruned.shape[1]]
    left_connection = np.mean(l_side)
    right_connection = np.mean(r_side)
    side_connection = (left_connection + right_connection) / 2
    return side_connection


def plot_fft_spectrum(img, prune_percentile=95):
    """
    Plot the FFT spectrum of the image, along with a high contrast version,
    and then along with a pruned version of this high contrast spectrum, in which
    only the values above the bottom {prune_percentile} are kept (e.g. at 95%,
    only the top 5%ile of values is displayed).
    Note that the image passed in should be the brightened/boosted/scaled Canny
    gradient image output, not an unmodified photo.
    """
    fig = plt.figure()
    ax1 = fig.add_subplot(3, 1, 1)
    imf = fft2(img)
    ax1.imshow(np.abs(imf), norm=LogNorm(vmin=5))
    ax2 = fig.add_subplot(3, 1, 2)
    ax2.imshow(img)
    ax3 = fig.add_subplot(3, 1, 3)
    mod_log_maxima = prune_fft(img, prune_percentile=prune_percentile)
    ax3.imshow(mod_log_maxima)
    plt.show()
    return


def prune_fft(img, prune_percentile=95, brighten_grade_img=False, boost_fft=False):
    """
    Take the 2D FFT of an image and prune values using given percentile.
    """
    if brighten_grade_img:
        img = BG_img(img)
    im_fft = fft2(img)
    if boost_fft:
        im_fft = brighten(boost_contrast(scale_img(np.log(np.abs(im_fft)))))
    else:
        im_fft = np.log(np.abs(im_fft))
    im_fft[np.where(im_fft < np.percentile(im_fft, prune_percentile))] = 0
    return im_fft


def show_max_pruned_cols(img, pruned_fft=None):
    """
    Display a summary of the pruned FFT from prune_fft into vertical bars
    representing the maximum value per column.
    """
    if pruned_fft is None:
        pruned = prune_fft(img)
    else:
        pruned = pruned_fft
    show_img(
        np.repeat(np.max(pruned.T, axis=1), pruned.T.shape[1]).reshape(pruned.T.shape).T
    )
    return
