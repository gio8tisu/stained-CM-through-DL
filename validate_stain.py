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

    if args.pad:
        # Pad scan with args.window_size // 2 on each side.
        scan = pad_image(scan, args.window_size // 2)

    cols = range(0, scan.width - args.window_size - 1, args.step)
    rows = range(0, scan.height - args.window_size - 1, args.step)
    result = {
        'homogeneity': np.empty((len(args.distances), len(args.angles),
                                 len(rows), len(cols))),
        'energy': np.empty((len(args.distances), len(args.angles),
                            len(rows), len(cols)))
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
                result['homogeneity'][..., row, col] = homogeneity
                result['energy'][..., row, col] = energy

                pbar.update()

    for feat in result:
        np.save(args.output + feat, result[feat])
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
    parser.add_argument('-o', '--output', help='output filename')
    parser.add_argument('--window-size', type=int, default=513, help='size in pixels of window')
    parser.add_argument('--step', type=int, help='Step size between windows')
    parser.add_argument('--pad', action='store_true', help='pad image with WINDOW_SIZE // 2 on each side')
    parser.add_argument('-v', '--verbose', action='store_true')
    subparsers = parser.add_subparsers(help='select method')
    glco_parser = subparsers.add_parser('glco', help='Extract grey-level co-occurrence matrix texture metrics')
    glco_parser.add_argument('--distances', nargs='+', type=int, default=[1, 2, 4])
    glco_parser.add_argument('--angles', nargs='+', type=float, default=[0, np.pi / 2, np.pi / 4])
    glco_parser.set_defaults(func=glco_main)
    ssim_parser = subparsers.add_parser('ssim', help='Compare digital stains with SSIM')
    ssim_parser.add_argument('--reference', required=True)
    ssim_parser.set_defaults(func=ssim_main)

    args = parser.parse_args()
    if args.verbose:
        print(args)

    # To transform from vips to numpy array.
    vips2numpy = Vips2Numpy.vips2numpy

    result = args.func(args)
