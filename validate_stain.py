import os.path
from itertools import product

import numpy as np
import pyvips
import tqdm
import skimage.feature.texture
from skimage.measure import compare_ssim

from numpy_pyvips import Vips2Numpy
from utils import pad_image


def glco_main(args):
    if args.step is None:
        args.step = 1

    # Read slide.
    scan = pyvips.Image.new_from_file(args.input)
    # Convert to grey scale.
    scan = scan.colourspace('b-w')
    # Repeat for reference image (if any).
    if args.reference:
        reference = pyvips.Image.new_from_file(args.reference)
        assert scan.height == reference.height and scan.width == reference.width
        reference = reference.colourspace('b-w')

    if args.pad:
        # Pad scan with args.window_size // 2 on each side.
        scan = pad_image(scan, args.window_size // 2)
        if args.reference:
            reference = pad_image(reference, args.window_size // 2)

    cols = range(0, scan.width - args.window_size - 1, args.step)
    rows = range(0, scan.height - args.window_size - 1, args.step)
    result = {
        'homogeneity': np.empty((len(rows), len(cols),
                                 len(args.distances), len(args.angles))),
        'energy': np.empty((len(rows), len(cols),
                            len(args.distances), len(args.angles)))
    }
    if args.reference:
        result_ref = {
            'homogeneity': np.empty((len(rows), len(cols),
                                     len(args.distances), len(args.angles))),
            'energy': np.empty((len(rows), len(cols),
                                len(args.distances), len(args.angles)))
        }

    with tqdm.tqdm(total=len(cols) * len(rows)) as pbar:
        for row, y_pos in enumerate(rows):
            for col, x_pos in enumerate(cols):
                # "Grab" window and apply pre-processing.
                array = preprocess(
                    scan.crop(x_pos, y_pos, args.window_size, args.window_size)
                )
                # Compute texture descriptors.
                P, homogeneity, energy = compute_comatrix_features(array,
                                                                   args.distances,
                                                                   args.angles)
                # Save features.
                result['homogeneity'][row, col] = homogeneity
                result['energy'][row, col] = energy
                # Repeat for reference image (if any).
                if args.reference:
                    array_ref = preprocess(
                        reference.crop(x_pos, y_pos,
                                       args.window_size, args.window_size)
                    )
                    P, homogeneity, energy = compute_comatrix_features(array_ref,
                                                                       args.distances,
                                                                       args.angles)
                    result_ref['homogeneity'][row, col] = homogeneity
                    result_ref['energy'][row, col] = energy

                pbar.update()

    for feat in result:
        if args.output:
            np.save(args.output + '-' + feat, result[feat])
        if args.reference:
            difference = result[feat] - result_ref[feat]
            print(feat, 'distance:', np.linalg.norm(difference))
            print(feat, 'average distance:',
                  np.mean(np.linalg.norm(difference, axis=(2, 3))))
            if args.output:
                np.save(args.output + '-' + feat + '_ref', result_ref[feat])
    return result


def lbp_main(args):
    if args.step is None:
        args.step = args.window_size

    # Read slide.
    scan = pyvips.Image.new_from_file(args.input)
    # Convert to grey scale.
    scan = scan.colourspace('b-w')
    # Repeat for reference image (if any).
    if args.reference:
        reference = pyvips.Image.new_from_file(args.reference)
        assert scan.height == reference.height and scan.width == reference.width
        reference = reference.colourspace('b-w')

    cols = range(0, scan.width - args.window_size - 1, args.step)
    rows = range(0, scan.height - args.window_size - 1, args.step)
    result = np.empty((len(rows), len(cols), 2 ** args.points))
    if args.reference:
        result_ref = np.empty((len(rows), len(cols), 2 ** args.points))

    with tqdm.tqdm(total=len(cols) * len(rows)) as pbar:
        for row, y_pos in enumerate(rows):
            for col, x_pos in enumerate(cols):
                # "Grab" window and apply pre-processing.
                array = preprocess(
                    scan.crop(x_pos, y_pos, args.window_size, args.window_size)
                )
                # Compute texture descriptors.
                lbp, hist = compute_lbp_histogram(array,
                                                  args.points, args.radius)
                # Save features.
                result[row, col] = hist
                # Repeat for reference image (if any).
                if args.reference:
                    array_ref = preprocess(
                        reference.crop(x_pos, y_pos,
                                       args.window_size, args.window_size)
                    )
                    lbp, hist = compute_lbp_histogram(array_ref,
                                                      args.points, args.radius)
                    result_ref[row, col] = hist

                pbar.update()

    if args.output:
        np.save(args.output, result)
    if args.reference:
        # Get function to compute histogram distances and compute them.
        dist_func = define_histogram_distance(args.histogram_distance)
        distances = dist_func(result, result_ref, axis=2)
        print('global distance:', np.linalg.norm(result - result_ref))
        print('average distance:', np.mean(distances))
        if args.output:
            np.save(args.output + '_ref', result_ref)
            # Write a file with window descriptions.
            with open(args.output + '_windows.csv', 'w') as f:
                print('crop-num', 'row', 'col',
                      'histogram-' + args.histogram_distance + '-distance',
                      sep=',', file=f)
                for n, (i, j) in enumerate(product(range(distances.shape[0]), range(distances.shape[1]))):
                    print(n, i, j, distances[i, j], sep=',', file=f)

        return result, result_ref
    return result


def ssim_main(args):
    if args.step is None:
        args.step = args.window_size

    # Read slide.
    scan = pyvips.Image.new_from_file(args.input)
    reference = pyvips.Image.new_from_file(args.reference)
    # Convert to grey scale.
    scan = scan.colourspace('b-w')
    reference = reference.colourspace('b-w')

    cols = range(0, scan.width - args.window_size - 1, args.step)
    rows = range(0, scan.height - args.window_size - 1, args.step)
    result = np.empty((len(rows), len(cols)))
    with tqdm.tqdm(total=len(cols) * len(rows)) as pbar:
        for row, y_pos in enumerate(rows):
            for col, x_pos in enumerate(cols):
                # "Grab" window and apply pre-processing.
                array = preprocess(
                    scan.crop(x_pos, y_pos, args.window_size, args.window_size)
                )
                array_ref = preprocess(
                    reference.crop(x_pos, y_pos,
                                   args.window_size, args.window_size)
                )
                # Compute texture descriptors.
                ssim = compare_ssim(array, array_ref)
                # Save.
                result[row, col] = ssim

                pbar.update()

    print('average SSIM:', np.mean(result))
    if args.output:
        np.save(args.output, result)
        # Write a file with window descriptions.
        with open(args.output + '_windows.csv', 'w') as f:
            f.write('row,col,ssim')
            print('row', 'col', 'ssim', sep=',', file=f)
            for i, j in product(range(result.shape[0]), range(result.shape[1])):
                print(i, j, result[i, j], sep=',', file=f)
    return result


def preprocess(crop):
    """Convert to numpy and cast to 8-bit unsigned integers."""
    array = vips2numpy(crop)
    # Scale to 0-255 range, cast to int8 and remove channel dimension.
    array = np.array(array * 255, dtype=np.uint8).squeeze()
    return array


def compute_comatrix_features(array, distances, angles):
    """Return co-occurrence matrix, homogeneity and energy."""
    # Compute co-occurrence matrix.
    P = skimage.feature.texture.greycomatrix(array, distances, angles,
                                             symmetric=True, normed=True)
    # Compute texture descriptors based on co-occurrence matrix.
    homogeneity = skimage.feature.texture.greycoprops(P, 'homogeneity')
    energy = skimage.feature.texture.greycoprops(P, 'energy')
    return P, homogeneity, energy


def compute_lbp_histogram(array, points, radius):
    """Return LBP image and normalized histogram."""
    # Compute LBP image.
    lbp = skimage.feature.texture.local_binary_pattern(array, points, radius)
    # Compute histogram.
    hist = np.bincount(lbp.flatten().astype(np.int), minlength=2 ** points)
    return lbp, hist / np.sum(hist)


def define_histogram_distance(distance_str):
    """Return function to compute distance between histograms.

    The returned function's arguments are hist_1, hist_2 and axis;
    where hist_1 and hist_2 are numpy arrays with histogram/s
    represented in the "axis" axis.
    """
    if distance_str.startswith('norm'):
        order = int(distance_str.split('-')[1])

        def dist(hist_1, hist_2, axis=None):
            difference = hist_1 - hist_2
            return np.linalg.norm(difference, order, axis)

    elif distance_str == 'chi-squared':

        def dist(hist_1, hist_2, axis=None):
            difference_squared = (hist_1 - hist_2) ** 2
            sum_ = hist_1 + hist_2
            return 0.5 * np.sum(np.true_divide(difference_squared, sum_,
                                               where=sum_ > 0),
                                axis=axis)

    elif distance_str == 'intersection':

        def dist(hist_1, hist_2, axis=None):
            minima = np.minimum(hist_1, hist_2)
            return 1 - minima.sum(axis=axis)

    else:
        raise NotImplementedError

    return dist


if __name__ == '__main__':
    # CLI arguments.
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input', type=str, help='path to input image')
    parser.add_argument('-o', '--output', required=False, help='output filename')
    parser.add_argument('--window-size', type=int, default=513, help='size in pixels of window')
    parser.add_argument('--step', type=int, help='Step size between windows')
    parser.add_argument('--pad', action='store_true', help='pad image with WINDOW_SIZE // 2 on each side')
    parser.add_argument('-v', '--verbose', action='store_true')
    subparsers = parser.add_subparsers(help='select method')
    glco_parser = subparsers.add_parser('glco', help='Extract grey-level co-occurrence matrix texture metrics')
    glco_parser.add_argument('--distances', nargs='+', type=int, default=[1, 2, 4])
    glco_parser.add_argument('--angles', nargs='+', type=float, default=[0, np.pi / 2, np.pi / 4])
    glco_parser.add_argument('--reference', required=False)
    glco_parser.set_defaults(func=glco_main)
    lbp_parser = subparsers.add_parser('lbp', help='Extract local binary pattern texture descriptor')
    lbp_parser.add_argument('--histogram-distance', default='chi-squared',
                            choices=[f'norm-{i}' for i in range(4)] + ['chi-squared', 'intersection'],
                            help='distance measure for histograms')
    lbp_parser.add_argument('--points', type=int, default=8, help='number of neighbors')
    lbp_parser.add_argument('--radius', type=int, default=1, help='radius of neighborhood')
    lbp_parser.add_argument('--reference', required=False)
    lbp_parser.set_defaults(func=lbp_main)
    ssim_parser = subparsers.add_parser('ssim', help='Compare digital stains with SSIM')
    ssim_parser.add_argument('--reference', required=True)
    ssim_parser.set_defaults(func=ssim_main)

    args = parser.parse_args()
    if args.verbose:
        print(args)

    # To transform from vips to numpy array.
    vips2numpy = Vips2Numpy.vips2numpy

    result = args.func(args)
