import os

import pyvips
import torch
import torchvision.transforms
import tqdm

import cyclegan.models
import numpy_pyvips
from utils import TileMosaic, pad_image
import transforms
from translate_wholeslides import transform_tile, normalize, scale


pyvips.voperation.cache_set_max_mem(100)
pyvips.voperation.cache_set_max_files(10)
# pyvips.voperation.cache_set_max(0)


def main(args):
    numpy2vips = numpy_pyvips.Numpy2Vips()

    # Read slide.
    slide_r = pyvips.Image.new_from_file(
        os.path.join(args.mosaic_directory, 'DET#1/highres_raw.tif'))
    slide_f = pyvips.Image.new_from_file(
        os.path.join(args.mosaic_directory, 'DET#2/highres_raw.tif'))
    # Slide pre-processing.
    if args.normalize:
        slide_r = normalize(slide_r)
        slide_f = normalize(slide_f)
    else:
        slide_r = scale(slide_r)
    slide_f = scale(slide_f)
    if args.normalization_method is not None:
        slide_r, slide_f = transforms.CMMinMaxNormalizer(
            args.normalization_method)(slide_r, slide_f)
    # Linear stain.
    scan = transforms.VirtualStainer()(slide_r, slide_f)
    # Define transform to apply patch-by-patch.
    transform = torchvision.transforms.Compose(
        [numpy_pyvips.Vips2Numpy(),
         torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
         ]
    )

    # Define model.
    G_AB = cyclegan.models.GeneratorResNet(res_blocks=9).to(device)
    # load model parameters using dataset_name and epoch number from CLI.
    G_AB.load_state_dict(torch.load(args.model, map_location=device))

    G_AB.eval()  # use evaluation/validation mode.
    size = args.patch_size
    crop = args.crop_size if args.crop_size else size // 2 + 1
    step = args.step if args.step else crop // 2

    tiles = TileMosaic(scan, size, crop,
                       0.25 if args.window == 'rectangular' else args.window)
    if args.verbose:
        print('Using {} as temporary directory'.format(tiles.tmp_dir.name))

    scan = pad_image(scan, size // 2)

    for x_pos in tqdm.trange(0, scan.width - size - 1, step):
        for y_pos in range(0, scan.height - size - 1, step):
            res = transform_tile(G_AB, numpy2vips, scan, size, transform, x_pos, y_pos, device)
            tiles.add_tile(res, x_pos, y_pos)
    transformed = tiles.get_mosaic()

    # Save result.
    transformed *= 255.0
    output_file = '{}.{}'.format(args.output, 'tif' if args.compression else args.format)
    if args.verbose:
        print('Saving transformed image to ' + output_file)
    if args.compression:
        transformed.tiffsave(output_file, tile=True, pyramid=True, compression='jpeg', Q=90)
    else:
        transformed.write_to_file(output_file)
    if args.save_linear:
        scan *= 255.0
        output_file = '{}_linear.{}'.format(args.output, 'tif' if args.compression else args.format)
        if args.verbose:
            print('Saving linear transform image to ' + output_file)
        if args.compression:
            scan.tiffsave(output_file, tile=True, pyramid=True, compression='jpeg', Q=90)
        else:
            scan.write_to_file(output_file)
    if args.verbose:
        print('Done.')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Transform CM whole-slide to H&E.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('mosaic_directory', type=str, help='path to mosaic directory')
    parser.add_argument('--model', required=True, help='path to model')
    parser.add_argument('-o', '--output', required=True, help='output file name without extension')
    format_group = parser.add_mutually_exclusive_group()
    format_group.add_argument('--format', default='jpg', help='output image format')
    format_group.add_argument('--compression', action='store_true', help='apply JPEG compression (with Q=90).')
    parser.add_argument('--patch-size', type=int, default=2048, help='size in pixels of patch/window.')
    parser.add_argument('--crop-size', type=int,
                        help='Crop size after transform, only used with --overlap option'
                             + ' (if None, fallback to PATCH_SIZE // 2 + 1)')
    parser.add_argument('--step', type=int,
                        help='Step size between patches. (if None, fallback to CROP_SIZE / 2')
    parser.add_argument('--save-linear', action='store_true',
                        help="save linearly stained image (input of model) to '*_linear_*'.")
    parser.add_argument('--window', default='rectangular',
                        help='window type for overlap option, should be and integer or one of: '
                             + 'rectangular, pyramid, circular or a number.')
    normalization_group = parser.add_mutually_exclusive_group()
    normalization_group.add_argument('--normalize', action='store_true',
                                     help='normalize slide (subtract min and divide by range)')
    normalization_group.add_argument('--normalization-method',
                                     choices=[None, 'independent', 'global', 'average'])
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--no-cuda', action='store_true', help='do not use GPU')

    args = parser.parse_args()
    if args.verbose:
        print(args)

    # check for correct --window argument.
    window_options = ['rectangular', 'pyramid', 'circular']
    if args.window not in window_options:
        try:
            args.window = float(args.window)
        except ValueError:
            print("'window' option should be one of "
                  + ' '.join(window_options) + ' or a real number.')
            exit(1)

    cuda = torch.cuda.is_available() and not args.no_cuda
    device = 'cuda:0' if cuda else 'cpu'

    with torch.no_grad():  # to avoid autograd overhead.
        main(args)
