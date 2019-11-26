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
        plt.imshow(trimmed_img, cmap=plt.get_cmap("gray"))
        for n, contour in enumerate(contours):
            plt.plot(contour[:, 1], contour[:, 0], linewidth=2)
        plt.show()
    return contours


def plot_contours(trimmed_img, contours):
    plt.imshow(trimmed_img, cmap=plt.get_cmap("gray"))
    for n, contour in enumerate(contours):
        plt.plot(contour[:, 1], contour[:, 0], linewidth=2)
    plt.show()
    return


def unique_unsorted(vals, axis=0):
    """
    Make a list of values unique by removing consecutive duplicates,
    under the condition that there will only be one run of consecutive
    duplicates in the list for each duplicated value.
    The axis provided is passed as np.unique(ar=vals, axis=axis).
    """
    # unique [sorted] list; indexes of [u for ul in vals]; ul counts
    # The 3 True parameters are to return: index, inverse, counts
    ul, ulj, uli, ulc = np.unique(vals, True, True, True, axis=axis)
    # Indexes of vals are denoted by i, indexes of ul are denoted by j
    # ulc_j_multi is a list of indexes on ul where count > 1
    ul_j_multi = np.where(ulc > 1)[0]
    ul_i_multi = [np.where([np.all(v == ul[j]) for v in vals])[0] for j in ul_j_multi]
    # The following line asserts that the duplicate values are consecutive together
    assert np.all([np.all(np.diff(r) == 1) for r in ul_i_multi])
    # To return a list that is unique, simply invert the reordering sort with argsort
    ul_presort = ul[ulj.argsort()]
    return ul_presort


def unpack_contours(contours):
    """
    Returns one rightward-oriented path per input contour in the list contours,
    so a list of 2 contours (closed polygons with repeated float values) is
    converted into a list of 2 paths from the leftmost to the rightmost (i.e.
    ascending x axis order), with all floats rounded and converted to integer
    arrays (outputs a list of numpy arrays, same as the input format).
    """
    paths = []
    # Iterate over the contours, sorted from leftmost to rightmost
    for c in sorted(contours, key=lambda x: min(x[1])):
        # All contour coords rounded to integer values now
        c_r = np.round(c).astype(int)
        c_min_x, c_max_x = np.min(c_r[:, 1]), np.max(c_r[:, 1])
        closed = np.array_equal(c_r[0], c_r[-1])
        assert closed, "I didn't code for unclosed contours (yet?)"
        leftward = c_r[0, 1] == c_max_x
        rightward = c_r[0, 1] == c_min_x
        assert leftward or rightward, "Contour is neither leftward or rightward?"
        if leftward:
            # This means the turning point is the leftmost x, c_min_x
            tp_i = np.where(c_r[:, 1] == c_min_x)[0]
        else:
            # This means the turning point is the rightmost x, c_max_x
            tp_i = np.where(c_r[:, 1] == c_max_x)[0]
        if tp_i.size > 1:
            # Check that the index values are consecutive
            assert np.all(np.diff(tp_i) == 1)
        outward = c_r[: tp_i[0] + 1]  # Ends on turning point
        inward = c_r[tp_i[-1] :]  # Starts at turning point
        in_u = unique_unsorted(inward, axis=0)
        out_u = unique_unsorted(outward, axis=0)
        symmetric = np.array_equal(in_u, out_u[::-1])
        if symmetric:
            if leftward:
                # Choose the rightward direction always, so here choose inward
                paths.append(in_u)
            else:
                # Choose the rightward direction always, so here choose outward
                paths.append(out_u)
        else:
            # For asymmetric outbound and inbound paths, quantify the deviation
            deviations = np.diff(list(zip(out_u, in_u[::-1])), axis=1)
            dx, dy = deviations[:, :, 1][:, 0], deviations[:, :, 0][:, 0]
            dev_x, dev_y = np.where(dx != 0)[0], np.where(dy != 0)[0]
            n_deviations = np.count_nonzero(deviations)
            assert n_deviations > 0, "The asymmetry isn't from deviated paths?"
            assert dev_x.size == 0, "The asymmetry is due to x axis deviations?"
            assert dev_y.size > 0, "Only asymmetry due to y axis deviation plz"
            # Just handle deviations on the y axis
            dev_y_coord = deviations[dev_y, 0, 0]
            # Probably caused by a boundary >1px thick, we want its bottom
            # Note that the contour finding algorithm (marching squares
            # implemented in skimage.measure.find_contours) goes counter-
            # clockwise because default parameter positive_orientation="low".
            # Counter-clockwise direction looks like clockwise for viewing
            # images as the y axis is flipped (try drawing a clock to see!)
            #
            # I don't think it's necessary to check each, just choose based
            # on which will be lower down the image (i.e. higher y value)
            if rightward:
                # inward y value is higher (bottom of boundary), choose that
                assert (
                    min(deviations[dev_y][:, :, 0]) > 0
                ), "A rightward path should wind counter-clockwise, deviation of +1"
                # Reverse a rightward inward path to make its direction rightward
                paths.append(in_u[::-1])
            else:
                # outward y value is higher (bottom of boundary), choose that
                assert (
                    min(deviations[dev_y][:, :, 0]) < 0
                ), "A leftward path should wind counter-clockwise, deviation of -1"
                # Reverse a leftward outward path to make its direction rightward
                paths.append(out_u[::-1])
    return paths


def join_paths(paths):
    """
    Takes a list of paths from `unpack_contours`, where each path is an integer
    numpy array of coordinates in order of (y,x), the order they are returned
    from skimage.measure.find_contours. These paths are expected to be listed
    in ascending x axis order, and checked for adjacency on the x axis.
    """
    adjacents = []
    joined = np.array([], dtype=np.int)
    for n, path in enumerate(paths):
        if n + 1 < len(paths):
            # Check if this path is adjacent to the next
            p1, p2 = paths[n : n + 2]
            y_diff, x_diff = np.diff([p1[-1], p2[0]], axis=0)[0]
            adjacent = x_diff == 1
            adjacents.append(adjacent)
    if np.all(adjacents):
        # Just join
        pass
    else:
        # Add an intermediate pixel? TODO: decide control logic
        pass
    return joined


def refine_contours(contours, x_win=(300, 800)):
    """
    Discard unnecessary contours from the contour set, by detecting overlap with the
    widest contour. Note that this could all be done on rounded coords but leave them
    as found since they are adjusted from int values for display purposes.
    """
    con_w = lambda c: np.max(c[:, 1] - np.min(c[:, 1]))
    contours_by_size = [c for c in reversed(sorted(contours, key=lambda x: len(x)))]
    contours_by_width = [c for c in reversed(sorted(contours, key=lambda x: con_w(x)))]
    contour_widths = [con_w(np.round(c)) for c in contours_by_width]
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
        both_ways = [[big_c, biggest_c], [biggest_c, big_c]]
        bridgeable = np.any(
            [np.max(a[:, 1] >= np.min(b[:, 1])) for (a, b) in both_ways]
        )
        # Only makes sense to combine two whose ranges overlap (bridgeable by isthmus)
        if con_w(biggest_c_r) + con_w(big_c_r) >= max_width - 1:
            print("Biggest two span the window")
            if bridgeable:
                print("The two are bridgeable")
                if con_w(biggest_c_r) + con_w(big_c_r) == max_width - 1:
                    print("The two are adjacent, not overlapping (bridge by joining)")
                    contour_segments = unpack_contours([biggest_c, big_c])
                    unified_path = join_paths(contour_segments)
                    return [unified_path]
                else:
                    print("The two are overlapping (bridge by merging)")
                return [biggest_c, big_c]
            else:
                print("These two can't be bridged though")
        else:
            print(f"Nah: biggest_c is {con_w(biggest_c)}, max_width is {max_width}")
    print(f"Non-overlap between the top 2 contours by size is {contour_diff.size}")
    return contours


def calculate_sparsity(
    img_n, crop_from_top=0.8, view=False, verbosity=1, x_win=(300, 800)
):
    """
    Calculate sparsities, after chopping off the top by a ratio
    of {crop_from_top} (e.g. 0.6 deducts 60% of image height).
    """
    print(f"Calculating sparsity for image {img_n}")
    im = imread(img_dir / images[img_n])
    crop_offset = round(im.shape[0] * crop_from_top)
    crop_im = im[crop_offset:, :, :]
    bg = BG_img(crop_im)
    sel, sparsities = scan_sparsity(bg, x_win=x_win, view=view, verbosity=verbosity)
    trimmed = bg[sel[0][0] : sel[0][1], x_win[0] : x_win[1]]
    return sel, sparsities, trimmed, crop_offset


def calculate_sparsities(crop_from_top=0.8, view=False, verbosity=1, x_win=(300, 800)):
    """
    Wrapper to calculate_sparsity, iterating over the full batch of images.
    """
    for img_n in range(0, len(images)):
        calc_results = calculate_sparsity(img_n, crop_from_top, view, verbosity, x_win)
        sel, sparsities, trimmed, crop_offset = calc_results
        del calc_results
        print()


def example_fft_plot():
    """
    Give an example usage of plot_fft_spectrum
    """
    img = imread(img_dir / images[0])
    bg = BG_img(img)
    plot_fft_spectrum(bg[3600:3700, 300:800])
    return


def example_scan_fft(crop_from_top=0.8, view=False, verbosity=1):
    """
    Give an example usage of scan_sparsity to give another FFT plot.
    """
    im = imread(img_dir / images[0])
    crop_im = im[round(im.shape[0] * crop_from_top) :, :, :]
    bg = BG_img(crop_im)
    x_win = (300, 800)  # Default window for scan_sparsity if not provided
    sel, sparsities = scan_sparsity(bg, view=view, verbosity=verbosity)
    sel_im = bg[sel[0][0] : sel[0][1], x_win[0] : x_win[1]]
    plot_fft_spectrum(sel_im)
    return


VISUALISE = True
# for img_n in range(0, len(images)):
for img_n in [10]:
    x_win = (300, 800)
    sel, sparsities, trimmed, y_offset = calculate_sparsity(
        img_n, view=VISUALISE, x_win=x_win
    )
    if None not in sel[0]:
        contours = get_contours(trimmed, view=VISUALISE)
        if len(contours) == 1:
            chosen_start, chosen_end = sel[0]
            print(
                f"✔ Contour found successfully in "
                + f"{y_offset+chosen_start}:{y_offset+chosen_end},{x_win[0]}:{x_win[1]}"
            )
            c = contours[0]
            c_min_x, c_max_x = np.min(c[:, 1]), np.max(c[:, 1])
            # N.B. only y values of countour points are jittered, so float equality
            # comparisons of contour coordinates' x values work without rounding
            assert c_max_x - c_min_x == abs(np.subtract(*x_win)) - 1.0
            c_min_y, c_max_y = np.round([np.min(c[:, 0]), np.max(c[:, 0])]).astype(int)
            win_y = y_offset + chosen_start
            contour_y_win = (win_y + c_min_y, win_y + c_max_y + 1)
            # NB: contour_y_win is a range, i.e. you stop 1 before its max. y
            print(
                f"More specifically, at {contour_y_win[0]}:{contour_y_win[1]},"
                + f"{x_win[0]}:{x_win[1]}"
            )
            bg_snippet = BG_img(
                imread(img_dir / images[img_n])[
                    contour_y_win[0] : contour_y_win[1], x_win[0] : x_win[1]
                ]
            )
            if VISUALISE:
                show_img(bg_snippet)
        elif len(contours) == 2 and img_n == 10:
            chosen_start, chosen_end = sel[0]
            print(
                f"2 contours found successfully in "
                + f"{y_offset+chosen_start}:{y_offset+chosen_end},{x_win[0]}:{x_win[1]}"
            )
            c1 = contours[0]
            c1_min_x, c1_max_x = np.min(c1[:, 1]), np.max(c1[:, 1])
            c2 = contours[1]
            c2_min_x, c2_max_x = np.min(c2[:, 1]), np.max(c2[:, 1])
            # N.B. only y values of countour points are jittered, so float equality
            # comparisons of contour coordinates' x values work without rounding
            # assert c_max_x - c_min_x == abs(np.subtract(*x_win)) - 1.0
            c1_min_y, c1_max_y = np.round([np.min(c1[:, 0]), np.max(c1[:, 0])]).astype(
                int
            )
            c2_min_y, c2_max_y = np.round([np.min(c2[:, 0]), np.max(c2[:, 0])]).astype(
                int
            )
            c12_min_y = min(c1_min_y, c2_min_y)
            c12_max_y = max(c1_max_y, c2_max_y)
            win_y = y_offset + chosen_start
            contour_y_win = (win_y + c12_min_y, win_y + c12_max_y + 1)
            # NB: contour_y_win is a range, i.e. you stop 1 before its max. y
            print(
                f"More specifically, at {contour_y_win[0]}:{contour_y_win[1]},"
                + f"{x_win[0]}:{x_win[1]}"
            )
            bg_snippet = BG_img(
                imread(img_dir / images[img_n])[
                    contour_y_win[0] : contour_y_win[1], x_win[0] : x_win[1]
                ]
            )
            if VISUALISE:
                show_img(bg_snippet)
    print()
# For images[0] (0th index i.e. 1st image in the list),
# the long contour is the 6th index (i.e. 7th in the list), 528 points long
# The 2nd longest is 522 points long (very close, indistinguishable on len alone)

# Assemble into individual PDFs

# authors: [420-422].jpg
# topics: [423-431].jpg
# symbols: 432.jpg
