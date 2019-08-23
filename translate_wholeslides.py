import os
from math import ceil

import pyvips
import numpy as np
import torch
import torchvision.transforms
import tqdm

import cyclegan.models
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
        scan = dataset[i]
        if args.verbose:
            print('Transforming image {} of {}'.format(i + 1, len(dataset)))
        image = None
        for x_pos in tqdm.trange(0, scan.width - size - 1, size):
            ver_image = None
            for y_pos in range(0, scan.height - size - 1, size):
                res = transform_tile(G_AB, numpy2vips, scan, size, transform, x_pos, y_pos, device)
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
        scan = dataset[i]

        if args.verbose:
            print('Transforming image {} of {}'.format(i + 1, len(dataset)))

        tiles = TileMosaic(scan, size, crop,
                           0.25 if args.window == 'rectangular' else args.window)
        scan = pad_image(scan, size // 2)
        if args.debug:
            x_count = 0
        for x_pos in tqdm.trange(0 if not args.debug else scan.width // 3,
                                 scan.width - size - 1, step):
            if args.debug and x_count > 5:
                break
            if args.debug:
                y_count = 0
            for y_pos in range(0 if not args.debug else scan.height // 3,
                               scan.height - size - 1, step):
                if args.debug and y_count > 5:
                    break
                res = transform_tile(G_AB, numpy2vips, scan, size, transform, x_pos, y_pos, device)
                tiles.add_tile(res, x_pos, y_pos)
                if args.debug:
                    y_count += 1
            if args.debug:
                x_count += 1
        image = tiles.get_mosaic()
        if args.save_linear:
            save(args, i, image, scan)
        else:
            save(args, i, image)
        if args.debug:
            break


def transform_tile(model, numpy2vips, scan, size, transform, x_pos, y_pos, device):
    tile_scan = scan.crop(x_pos, y_pos, size, size)  # "grab" square window/patch from image.
    tile_scan = transform(tile_scan)  # convert to torch tensor and channels first.
    tile_scan = tile_scan.to(device)
    res = model(tile_scan.unsqueeze(0))  # reshape first for batch axis.
    res_np = res.detach().to('cpu').numpy()  # get data as numpy array.
    res_np = np.moveaxis(res_np, 1, 3)  # to channels last.
    res_np = (res_np[0] + 1) / 2  # shift pixel values to [0,1] range.
    res_vips = numpy2vips(res_np)  # convert to pyvips.Image
    return res_vips


def save(args, i, transformed, linear=None):
    transformed *= 255.0
    output_file = os.path.join(args.output, '{}{}.{}'.format(args.prefix, i, 'tif' if args.compression else args.format))
    if args.verbose:
        print('Saving transformed image to ' + output_file)
    if args.compression:
        # transformed.tiffsave(output_file, tile=True, pyramid=True, compression='deflate', Q=90)
        transformed.tiffsave(output_file, tile=True, pyramid=True, compression='jpeg', Q=90)
    else:
        transformed.write_to_file(output_file)
    if linear:
        linear *= 255.0
        output_file = os.path.join(args.output, '{}_linear_{}.{}'.format(args.prefix, i, 'tif' if args.compression else args.format))
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


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Transform CM whole-slides to H&E.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('data_directory', type=str, help='directory with mosaic* directories')
    parser.add_argument('--models-dir', required=True, help='directory with saved models')
    parser.add_argument('-o', '--output', required=True, help='output directory')
    parser.add_argument('--prefix', default='scan', help='output files prefix PREFIX')
    format_group = parser.add_mutually_exclusive_group()
    format_group.add_argument('--format', default='jpg', help='output image format')
    format_group.add_argument('--compression', action='store_true', help='apply JPEG compression (with Q=90).')
    parser.add_argument('--epoch', type=int, default=199, help='epoch to get model from.')
    parser.add_argument('--patch-size', type=int, default=2048, help='size in pixels of patch/window.')
    parser.add_argument('--crop-size', type=int,
                        help='Crop size after transform, only used with --overlap option'
                             + ' (if None, fallback to PATCH_SIZE // 2 + 1)')
    parser.add_argument('--step', type=int,
                        help='Step size between patches. (if None, fallback to CROP_SIZE / 2')
    parser.add_argument('--dataset-name', type=str, default='conf_data6', help='name of the saved model dataset.')
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

    numpy2vips = numpy_pyvips.Numpy2Vips()

    # transform to apply patch-by-patch.
    patch_transform = torchvision.transforms.Compose(
        [numpy_pyvips.Vips2Numpy(),
         torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
         ]
    )
    # dataset of confocal large slides.
    dataset = SkinCMDataset(
        args.data_directory, stain=True,
        transform_F=torchvision.transforms.Lambda(normalize if args.normalize else scale),
        transform_R=torchvision.transforms.Lambda(normalize if args.normalize else scale),
        transform=transforms.CMMinMaxNormalizer(args.normalization_method) if args.normalization_method else None
    )

    G_AB = cyclegan.models.GeneratorResNet(res_blocks=9)
    if cuda:
        G_AB = G_AB.cuda()
    # load model parameters using dataset_name and epoch number from CLI.
    G_AB.load_state_dict(torch.load(
        os.path.join(args.models_dir, args.dataset_name, f'G_AB_{args.epoch}.pth'),
        map_location=device
    ))

    G_AB.eval()  # use evaluation/validation mode.
    with torch.no_grad():  # to avoid autograd overhead.
        if args.overlap:
            main_fancy(args, dataset, G_AB, patch_transform, numpy2vips)
        else:
            main(args, dataset, G_AB, patch_transform, numpy2vips)
