import os
from math import ceil

import pyvips
import numpy as np
import torch
import torchvision.transforms
import tqdm

import numpy_pyvips
from datasets import SkinCMDataset
from utils import TileMosaic, pad_image
import transforms


pyvips.voperation.cache_set_max_mem(100)
pyvips.voperation.cache_set_max_files(10)
# pyvips.voperation.cache_set_max(0)


def main(args, dataset, G_AB, transform, numpy2vips):
    size = args.patch_size
    for i in range(len(dataset)):
        scan, prefix = dataset[i]
        if args.verbose:
            print('Transforming image {} of {} ({})'.format(i + 1, len(dataset), prefix))
        image = None
        for x_pos in tqdm.trange(0, scan.width - size - 1, size):
            ver_image = None
            for y_pos in range(0, scan.height - size - 1, size):
                res = transform_tile(G_AB, scan, size, transform, x_pos, y_pos, device, numpy2vips)
                ver_image = res if not ver_image else ver_image.join(res, 'vertical')  # "stack" vertically
            image = ver_image if not image else image.join(ver_image, 'horizontal')  # "stack" horizontally
        if args.save_linear:
            save(args, i, image, scan)
        else:
            save(args, i, image)


def main_fancy(args, dataset, G_AB, transform, numpy2vips):
    size = args.patch_size
    crop = args.crop_size if args.crop_size else size // 2 + 1
    step = args.step if args.step else crop // 2
    for i in range(len(dataset)):
        scan, prefix = dataset[i]
        if args.verbose:
            print('Transforming image {} of {} ({})'.format(i + 1, len(dataset), prefix))

        tiles = TileMosaic(scan, 0.25 if args.window == 'rectangular' else args.window, size, crop,
                           pyvips_tiles=not args.no_pyvips_tiles)
        if args.verbose:
            print('Using {} as temporary directory'.format(tiles.tmp_dir.name))
        scan = pad_image(scan, size // 2)
        if args.debug:
            y_count = 0
        with tqdm.tqdm() as pbar:
            y_pos = 0
            while y_pos < scan.height - size - 1:
                x_pos = 0
                if args.debug and y_count > 5:
                    break
                if args.debug:
                    x_count = 0
                while x_pos < scan.width - size - 1:
                    if args.debug and y_count > 5:
                        break
                    res = transform_tile(G_AB, scan, size, transform, x_pos, y_pos, device, numpy2vips)
                    tiles.add_tile(res, x_pos, y_pos)
                    x_pos += step
                    pbar.update()
                    if args.debug:
                        x_count += 1
                y_pos += step
                if args.debug:
                    y_count += 1
        image = tiles.get_mosaic()
        if args.save_linear:
            save(args, i, image, scan)
        else:
            save(args, i, image)
        if args.debug:
            break


def transform_tile(model, scan, size, transform, x_pos, y_pos, device, numpy2vips=None):
    tile_scan = scan.crop(x_pos, y_pos, size, size)  # "grab" square window/patch from image.
    tile_scan = transform(tile_scan)  # convert to torch tensor and channels first.
    tile_scan = tile_scan.to(device)
    res = model(tile_scan.unsqueeze(0))  # reshape first for batch axis.
    res = res.detach().to('cpu').numpy()  # get data as numpy array.
    res = np.moveaxis(res, 1, 3)  # to channels last.
    res = (res[0] + 1) / 2  # shift pixel values to [0,1] range.
    if numpy2vips is not None:
        res = numpy2vips(res)  # convert to pyvips.Image
    return res


def save(args, i, transformed, linear=None):
    transformed *= 255.0
    output_file = os.path.join(
        args.output,
        '{}{}.{}'.format(args.prefix, i, 'tif' if args.compression else args.format)
    )
    if args.verbose:
        print('Saving transformed image to ' + output_file)
    if args.compression:
        # transformed.tiffsave(output_file, tile=True, pyramid=True, compression='deflate', Q=90)
        transformed.tiffsave(output_file, tile=True, pyramid=True, compression='jpeg', Q=90)
    else:
        transformed.write_to_file(output_file)
    if linear:
        linear *= 255.0
        output_file = os.path.join(
            args.output,
            '{}{}_linear.{}'.format(args.prefix, i, 'tif' if args.compression else args.format)
        )
        if args.verbose:
            print('Saving linear transform image to ' + output_file)
        if args.compression:
            # linear.tiffsave(output_file, tile=True, pyramid=True, compression='deflate', Q=90)
            linear.tiffsave(output_file, tile=True, pyramid=True, compression='jpeg', Q=90)
        else:
            linear.write_to_file(output_file)
    if args.verbose:
        print('Done.')


def normalize(img):
    """Normalize pyvips.Image by min and max.

    Intended to be used with torchvision.transforms.Lambda
    """
    min_ = img.min()
    range_ = img.max() - min_
    return (img - min_) / range_


def scale(img, s=65535):
    """Scale pyvips.Image by s.

    Intended to be used with torchvision.transforms.Lambda
    """
    return img / s


def get_affine_model(args):
    import affine_cyclegan
    return affine_cyclegan.AffineGenerator(2, 3)


def get_resnet_model(args):
    import cyclegan.models
    return cyclegan.models.GeneratorResNet(res_blocks=args.n_blocks)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Transform CM whole-slides to H&E.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('data_directory', type=str, help='directory with mosaic* directories')
    parser.add_argument('--model-path', required=True, help='directory with saved models')
    parser.add_argument('-o', '--output', required=True, help='output directory')
    parser.add_argument('--prefix', default='scan', help='output files prefix PREFIX')
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
    parser.add_argument('--overlap', action='store_true',
                        help='overlapping tiles (WSI inference technique by Thomas de Bel et al.)')
    parser.add_argument('--window', default='rectangular',
                        help='window type for overlap option, should be and integer or one of: '
                             + 'rectangular, pyramid, circular or a number.')
    normalization_group = parser.add_mutually_exclusive_group()
    normalization_group.add_argument('--normalize', action='store_true',
                                     help='normalize slide (subtract min and divide by range)')
    normalization_group.add_argument('--normalization-method',
                                     choices=[None, 'independent', 'global', 'average'])
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--no-cuda', action='store_true', help='do not use GPU')
    parser.add_argument('--no-pyvips-tiles', action='store_true', help='do not use pyvips in tile creation')
    subparsers = parser.add_subparsers(title='model-type', dest='model_type')
    affine_parser = subparsers.add_parser('affine')
    affine_parser.set_defaults(get_model=get_affine_model)
    resnet_parser = subparsers.add_parser('resnet')
    resnet_parser.set_defaults(get_model=get_resnet_model)
    resnet_parser.add_argument('--n-blocks', type=int, default=9,
                               help='number of residual blocks in generator')

    args = parser.parse_args()
    if args.verbose:
        print(args)

    # check if output dir exists.
    if not os.path.isdir(args.output):
        if args.verbose:
            print('Output directory does not exist, creating it...')
        os.makedirs(args.output)

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

    numpy2vips = numpy_pyvips.Numpy2Vips() if not args.no_pyvips_tiles else None

    # transform to apply patch-by-patch.
    if args.model_type == 'resnet':
        patch_transform = torchvision.transforms.Compose(
            [numpy_pyvips.Vips2Numpy(),
             torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
             ]
        )
    else:
        pass
    if args.model_type == 'resnet':
        # dataset of confocal large slides.
        dataset = SkinCMDataset(
            args.data_directory, stain=True,
            transform_F=torchvision.transforms.Lambda(normalize if args.normalize else scale),
            transform_R=torchvision.transforms.Lambda(normalize if args.normalize else scale),
            transform=transforms.CMMinMaxNormalizer(args.normalization_method) if args.normalization_method else None,
            return_prefix=True
        )
    else:
        pass

    G_AB = args.get_model(args)
    if cuda:
        G_AB = G_AB.cuda()
    # load model parameters using dataset_name and epoch number from CLI.
    G_AB.load_state_dict(torch.load(args.model_path, map_location=device))

    G_AB.eval()  # use evaluation/validation mode.
    with torch.no_grad():  # to avoid autograd overhead.
        if args.overlap:
            main_fancy(args, dataset, G_AB, patch_transform, numpy2vips)
        else:
            main(args, dataset, G_AB, patch_transform, numpy2vips)
