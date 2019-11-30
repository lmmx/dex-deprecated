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


def get_contours(
    trimmed_img, view=False, level=254, connectedness="high", refine=True, verbosity=1
):
    """
    Pass in a trimmed gradient image (e.g. from `calculate_sparsity`) and get
    a list of contours found in that region, which can then be processed to
    find a single contour. This function is a user-friendly or 'front facing'
    version of contour_line() in `image_funcs.py`.
    """
    contours = find_contours(trimmed_img, level=level, fully_connected=connectedness)
    if refine:
        contours = refine_contours(contours, verbosity=verbosity)
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


def close_contour(contour, verbose=True):
    assert contour.dtype == np.int, "contour is not an integer coordinate array"
    start, end = contour[0], contour[-1]
    if np.array_equal(start, end):
        return contour  # contour is already closed
    c_min_x, c_max_x = np.min(contour[:, 1]), np.max(contour[:, 1])
    extreme_n = np.where(np.in1d([start[1], end[1]], [c_min_x, c_max_x]))[0][0]
    if extreme_n == 0:
        # The start point is the extreme (extreme_n = 0), so extend the end
        shared_i = np.where([np.array_equal(end, c) for c in contour])[0][0]
        extension = contour[:shared_i][::-1]
        contour = np.vstack([contour, extension])
    else:
        assert extreme_n == 1
        # The end point is the extreme (extreme_n = 1), so extend the start
        shared_i = np.where([np.array_equal(start, c) for c in contour[::-1]])[0][0]
        extension = contour[::-1][:shared_i]
        contour = np.vstack([extension, contour])
    closed = np.array_equal(contour[0], contour[-1])
    if verbose:
        print(f"Contour closed by extending {len(extension)}px (shared_i={shared_i})")
    assert closed, "This contour couldn't be closed."
    return contour


def unpack_contours(contours, verbosity=1):
    """
    Returns one rightward-oriented path per input contour in the list contours,
    so a list of 2 contours (closed polygons with repeated float values) is
    converted into a list of 2 paths from the leftmost to the rightmost (i.e.
    ascending x axis order), with all floats rounded and converted to integer
    arrays (outputs a list of numpy arrays, same as the input format).

    Verbosity level 0 will print nothing.
    Verbosity level 1 will print only that a contour was found successfully.
    Verbosity level 2 will print debuggable contour info (progress and coords).
    """
    paths = []
    # Iterate over the contours, sorted from leftmost to rightmost
    for c in sorted(contours, key=lambda x: min(x[1])):
        if verbosity > 1:
            print("Processing a contour")
        # All contour coords rounded to integer values now
        c_r = np.round(c).astype(int)
        closed = np.array_equal(c_r[0], c_r[-1])
        if not closed:
            if verbosity > 1:
                print("Closing contour...")
            # Extend the shorter end to close it
            c_r = close_contour(c_r, verbose=(verbosity > 1))
        c_min_x, c_max_x = np.min(c_r[:, 1]), np.max(c_r[:, 1])
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
            if verbosity > 1:
                print(len(c_r))
            assert np.all(np.diff(tp_i) == 1), f"Not consecutive: {tp_i}"
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
    from skimage.measure.find_contours, and joins them at their junction(s),
    adjusting the y values of terminal points (+1 or -1) as necessary, then
    returning one single array from concatenating the listed input paths. These
    input paths are expected to be listed in ascending x axis order, and are
    checked for adjacency on the x axis (where adjacency means their terminal
    points' x values differ by 1), otherwise throwing a ValueError which may
    indicate `merge_paths` should have been used instead.
    """
    adjacents = []
    for N in range(len(paths) - 1):
        # Check if this path is adjacent to the next (the junction)
        junc = [p[n - 1] for n, p in enumerate(paths[N : N + 2])]
        x_diff = np.diff(junc, axis=0)[0, 1]
        adjacents.append(x_diff == 1)  # True if extremal x values differ by 1
    if np.all(adjacents):
        # Just join
        for N in range(len(paths) - 1):
            junc = [p[n - 1] for n, p in enumerate(paths[N : N + 2])]
            y_diff = np.diff(junc, axis=0)[0, 0]
            # Also take the penultimate end p1[-2] and second start p2[1] values:
            junc2 = [p[3 * n - 2] for n, p in enumerate(paths[N : N + 2])]
            # y_diff from p1[-1] to p1[-2] & p2[0] to p2[1] (i.e. away from junction)
            p1_j2_dy, p2_j2_dy = np.diff([junc, junc2], axis=0)[0][:, 0]
            # If p1[-1] is > 1 pixels higher up (smaller y), lower it (increase y) by 1
            # if it'll stay 8-connected to the previous [penultimate] point, stored in
            # junc2, y_diff of p1_j2_dy (vice versa for lowering p2[0] if p2[1]... etc)
            if abs(y_diff) > 1:
                # There is a disconnected step (i.e. 2 pixel y gap, not 8-connected)
                assert abs(y_diff) == 2, f"The gap to bridge is too big: {y_diff} px"
                # could handle a gap of up to 4, but requires 2nd check (j3_dy)
                if y_diff > 0:
                    # There is a positive y + 2 step (i.e. rightwards, down page)
                    # y_diff = +2
                    # If y_diff backward on p1 counteracts y_diff across the junction
                    if p1_j2_dy >= 0:
                        # p1[-2] will stay connected if you lower p1[-1]
                        paths[N][-1, 0] += 1  # lower p1[-1] (increment y value)
                        pass
                    elif p2_j2_dy <= 0:
                        # p2[1] will stay connected if you raise p2[0]
                        paths[N + 1][0, 0] -= 1  # raise p2[0] (decrement y value)
                        pass
                    else:
                        raise ValueError("Gap unbridgeable due to divergent junction")
                else:
                    # There is a negative y - 2 step (i.e. rightwards, up page)
                    # y_diff = -2
                    # If y_diff across the junction minus y_diff forward on p1
                    if p1_j2_dy <= 0:
                        # p1[-2] will stay connected if you raise p1[-1]
                        paths[N][-1, 0] -= 1  # raise p1[-1] (decrement y value)
                        pass
                    elif p2_j2_dy >= 0:
                        # p2[1] will stay connected if you lower p2[0]
                        paths[N + 1][0, 0] += 1  # lower p2[0] (inrement y value)
                        pass
                    else:
                        raise ValueError("Gap unbridgeable due to divergent junction")
        joined = np.vstack(paths)
    else:
        # Add an intermediate pixel? Case of > 1 px x axis distance is not handled
        raise ValueError("Sorry I've not yet coded for the non-adjacent case")
    return joined


def merge_paths(paths):
    """
    Takes a list of paths from `unpack_contours`, where each path is an integer
    numpy array of coordinates in order of (y,x), the order they are returned
    from skimage.measure.find_contours, and joins them at their junction(s),
    which are expected to overlap by one or more coordinates on the x axis, by
    identifying the 'run' or 'plateau' for each of the overlapping terminal points
    (i.e. the set of points extending away from the overlapping region with the
    same y value as its terminal point) then identifying the first change in y
    value that takes place at the end of this 'run'/'plateau' of points), which
    then is used to decide which of the two plateaus are extensible (possibly both,
    in which case the lower of the two is chosen to be extended, to ensure the
    bottom-most boundary is used) returning one single array from concatenating
    the listed input paths. These input paths are expected to be listed in
    ascending x axis order, and are checked for overlap on the x axis (where
    overlap means one or more of their terminal points' x values match), otherwise
    throwing a ValueError, which may indicate `join_paths` should have been used
    instead.
    """
    for N in range(len(paths) - 1):
        # Check if this path is adjacent to the next (the junction)
        junc = [p[n - 1] for n, p in enumerate(paths[N : N + 2])]
        x_diff = np.diff(junc, axis=0)[0, 1]
        assert x_diff < 0, f"Paths do not seem to overlap: x_diff = {x_diff}"
        y_diff = np.diff(junc, axis=0)[0, 0]
        assert y_diff != 0, "Overlap ends share y coord.: just fuse the junction"
        # Get the y_diff per point transition away from the junction on both sides
        p1r_y = paths[N][:, 0][::-1]
        p2_y = paths[N + 1][:, 0]
        # Get the non-0 y_diff per point transition away from junction 1
        j1_dy = p1r_y[(j1_dy_i := np.where(np.diff(p1r_y) != 0)[0])]
        # Get the non-0 y_diff per point transition away from junction 1
        j2_dy = p2_y[(j2_dy_i := np.where(np.diff(p2_y) != 0)[0])]
        # Get the first y transition away from the junction for p1 and p2
        j1_dy_1 = np.diff([j1_dy[0], p1r_y[j1_dy_i[0] + 1]])[0]
        j2_dy_1 = np.diff([j2_dy[0], p2_y[j2_dy_i[0] + 1]])[0]
        if y_diff > 0:
            # Handle j1 being higher (smaller y) and j2 being lower (larger y)
            if j1_dy_1 > 0:
                # It matches the change in y direction to p2, so merge to p2_y
                end_plateau = paths[N][::-1][: j1_dy_i[0] + 1]
                # Shift the end plateau (of p1) downward by y_diff pixels
                end_plateau[:, 0] += y_diff
                # Finally, trim the end plateau to eliminate overlap with p2
                paths[N] = paths[N][: x_diff - 1]
            elif j2_dy_1 < 0:
                # Resort to shifting p2 up since p1 cannot be shifted down
                end_plateau = paths[N + 1][: j2_dy_i[0] + 1]
                # Shift p2 upward by y_diff pixels to merge it with p1
                end_plateau[:, 0] -= y_diff
                # Finally, trim the end plateau to eliminate overlap with p1
                paths[N + 1] = paths[N + 1][1 - x_diff :]
            else:
                # If this happens, add more edge case handling, raise error for now
                raise ValueError("Reached a junction which diverges, can't merge")
        else:
            # Handle j1 being lower (larger y) and j2 being higher (smaller y)
            if j2_dy_1 > 0:
                # It matches the change in y direction to p2, so merge to p2_y
                end_plateau = paths[N + 1][: j2_dy_i[0] + 1]
                # Shift p2 upward by y_diff pixels to merge it with p1
                end_plateau[:, 0] -= y_diff
                # Finally, trim the end plateau to eliminate overlap with p1
                paths[N + 1] = paths[N + 1][1 - x_diff :]
            elif j1_dy_1 < 0:
                # Resort to shifting p1 up since p2 cannot be shifted down
                end_plateau = paths[N][::-1][: j1_dy_i[0] + 1]
                # Shift the end plateau (of p1) downward by y_diff pixels
                end_plateau[:, 0] += y_diff
                # Finally, trim the end plateau to eliminate overlap with p2
                paths[N] = paths[N][: x_diff - 1]
            else:
                # If this happens, add more edge case handling, raise error for now
                raise ValueError("Reached a junction which diverges, can't merge")
    merged = np.vstack(paths)
    return merged


def refine_contours(contours, x_win=(300, 800), verbosity=1):
    """
    Discard unnecessary contours from the contour set, by detecting overlap with the
    widest contour. Note that this could all be done on rounded coords but leave them
    as found since they are adjusted from int values for display purposes.
    """
    if verbosity < 1:
        verbose = False
        v_v = False
    elif verbosity > 1:
        verbose = True
        v_v = True
    elif verbosity == 1:
        verbose = True
        v_v = False
    else:
        raise ValueError("Bad verbosity value, try an integer from -1 to 2")
    con_w = lambda c: np.max(c[:, 1] - np.min(c[:, 1]))
    contours_by_size = [c for c in reversed(sorted(contours, key=lambda x: len(x)))]
    contours_by_width = [c for c in reversed(sorted(contours, key=lambda x: con_w(x)))]
    contour_widths = [con_w(np.round(c)) for c in contours_by_width]
    if v_v:
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
        if v_v:
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
            if v_v:
                print("Biggest two span the window")
            if bridgeable:
                if v_v:
                    print("The two are bridgeable")
                contour_segments = unpack_contours([biggest_c, big_c])
                if con_w(biggest_c_r) + con_w(big_c_r) == max_width - 1:
                    if v_v:
                        print("The two are adjacent, not overlapping (joinable)")
                    unified_path = join_paths(contour_segments)
                else:
                    if v_v:
                        print("The two are overlapping on the x axis (mergeable)")
                    unified_path = merge_paths(contour_segments)
                return [unified_path]
            elif v_v:
                print("These two can't be bridged though")
        elif v_v:
            print(f"Nah: biggest_c is {con_w(biggest_c)}, max_width is {max_width}")
    if v_v:
        print(f"Non-overlap between top 2 contours by size is {contour_diff.size}")
    return contours


def calculate_sparsity(
    img_n, crop_from_top=0.8, view=False, verbosity=1, x_win=(300, 800)
):
    """
    Calculate sparsities, after chopping off the top by a ratio
    of {crop_from_top} (e.g. 0.6 deducts 60% of image height).
    """
    if verbosity > 1:
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
        if verbosity > -1: print()


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


def bound_all_pages(
    img_n=None, view=False, view_grad=False, x_win_init=None, verbosity=1
):
    if img_n is None:
        img_n_set = [n for n in range(0, len(images))]
    elif len(img_n) > 1:
        img_n_set = img_n
    elif len(img_n) == 1:
        img_n_set = [img_n]
    else:
        raise ValueError(f"img_n parameter {img_n} is not > 0")
    for img_n in img_n_set:
        init_bound_page(img_n, view, view_grad, x_win_init, verbosity)
        if verbosity > 0: print()
    return


def init_bound_page(img_n, view=False, view_grad=False, x_win_init=None, verbosity=1):
    if x_win_init is None:
        x_win = (300, 800)
    else:
        x_win = x_win_init
    sel, sparsities, trimmed, y_offset = calculate_sparsity(
        img_n, view=view, x_win=x_win, verbosity=verbosity
    )
    if None in sel[0]:
        return
    contours = get_contours(trimmed, view=view, verbosity=verbosity)
    if len(contours) == 1:
        chosen_start, chosen_end = sel[0]
        if verbosity > 0:
            print(
                f"✔ Contour found successfully in {y_offset+chosen_start}:"
                + f"{y_offset+chosen_end},{x_win[0]}:{x_win[1]}"
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
        if verbosity > 1:
            print(
                f"More specifically, at {contour_y_win[0]}:{contour_y_win[1]},"
                + f"{x_win[0]}:{x_win[1]}"
            )
        bg_snippet = BG_img(
            imread(img_dir / images[img_n])[
                contour_y_win[0] : contour_y_win[1], x_win[0] : x_win[1]
            ]
        )
        if view_grad:
            show_img(bg_snippet)
    elif len(contours) == 2:
        chosen_start, chosen_end = sel[0]
        if verbosity > 1:
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
        c1_min_y, c1_max_y = np.round([np.min(c1[:, 0]), np.max(c1[:, 0])]).astype(int)
        c2_min_y, c2_max_y = np.round([np.min(c2[:, 0]), np.max(c2[:, 0])]).astype(int)
        c12_min_y = min(c1_min_y, c2_min_y)
        c12_max_y = max(c1_max_y, c2_max_y)
        win_y = y_offset + chosen_start
        contour_y_win = (win_y + c12_min_y, win_y + c12_max_y + 1)
        # NB: contour_y_win is a range, i.e. you stop 1 before its max. y
        if verbosity > 1:
            print(
                f"More specifically, at {contour_y_win[0]}:{contour_y_win[1]},"
                + f"{x_win[0]}:{x_win[1]}"
            )
        bg_snippet = BG_img(
            imread(img_dir / images[img_n])[
                contour_y_win[0] : contour_y_win[1], x_win[0] : x_win[1]
            ]
        )
        if view_grad:
            show_img(bg_snippet)
    return


# For images[0] (0th index i.e. 1st image in the list),
# the long contour is the 6th index (i.e. 7th in the list), 528 points long
# The 2nd longest is 522 points long (very close, indistinguishable on len alone)

# Assemble into individual PDFs

# authors: [420-422].jpg
# topics: [423-431].jpg
# symbols: 432.jpg
