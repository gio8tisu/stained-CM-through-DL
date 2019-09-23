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
                P, homogeneity, energy = compute_features(array,
                                                          args.distances,
                                                          args.angles)
                # Save features.
                result['homogeneity'][row, col] = homogeneity
                result['energy'][row, col] = energy
                # Repeat for reference image (if any).
                if args.reference:
                    array_ref = preprocess(
                        reference.crop(x_pos, y_pos, args.window_size, args.window_size)
                    )
                    P, homogeneity, energy = compute_features(array_ref,
                                                              args.distances,
                                                              args.angles)
                    result_ref['homogeneity'][row, col] = homogeneity
                    result_ref['energy'][row, col] = energy

                pbar.update()

    for feat in result:
        np.save(args.output + '-' + feat, result[feat])
        if args.reference:
            print(feat, 'distance:', np.linalg.norm(result[feat] - result_ref[feat]))
            np.save(args.output + '-' + feat + '_ref', result_ref[feat])
    return result


def lbp_main(args):
    scan = pyvips.Image.new_from_file(args.input)
    scan = scan.colourspace('b-w')

    result = skimage.feature.texture.local_binary_pattern(preprocess(scan),
                                                          args.points,
                                                          args.radius)

    np.save(args.output, result)
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
                    reference.crop(x_pos, y_pos, args.window_size, args.window_size)
                )
                # Compute texture descriptors.
                ssim = compare_ssim(array, array_ref)
                # Save.
                result[row, col] = ssim

                pbar.update()

    np.save(args.output, result)
    return result


def preprocess(crop):
    """Convert to numpy and cast to 8-bit unsigned integers."""
    array = vips2numpy(crop)
    # Scale to 0-255 range, cast to int8 and remove channel dimension.
    array = np.array(array * 255, dtype=np.uint8).squeeze()
    return array


def compute_features(array, distances, angles):
    """Return co-occurrence matrix, homogeneity and energy."""
    # Compute co-occurrence matrix.
    P = skimage.feature.texture.greycomatrix(array, distances, angles,
                                             symmetric=True, normed=True)
    # Compute texture descriptors based on co-occurrence matrix.
    homogeneity = skimage.feature.texture.greycoprops(P, 'homogeneity')
    energy = skimage.feature.texture.greycoprops(P, 'energy')
    return P, homogeneity, energy


if __name__ == '__main__':
    # CLI arguments.
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input', type=str, help='path to input image')
    parser.add_argument('-o', '--output', required=True, help='output filename')
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
