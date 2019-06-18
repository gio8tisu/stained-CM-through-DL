import os

from PIL import Image
import pyvips
import numpy as np
import torch
from torchvision import transforms
import tqdm

import cyclegan.models
import numpy_pyvips
from datasets import SkinDataset
from utils import TileMosaic, pad_image


pyvips.cache_set_max(0)

Image.MAX_IMAGE_PIXELS = None


def main(args, dataset, G_AB, transform, numpy2vips, cuda):
    size = args.patch_size
    for i in range(len(dataset)):
        scan = dataset[i]
        if args.verbose:
            print('Transforming image {} of {}'.format(i + 1, len(dataset)))
        image = None
        for x_pos in tqdm.trange(0, scan.width - size - 1, size):
            ver_image = None
            for y_pos in range(0, scan.height - size - 1, size):
                tile_scan = scan.crop(x_pos, y_pos, size, size)  # "grab" square window/patch from image.
                tile_scan = transform(tile_scan)  # convert to torch tensor and channels first.
                if cuda:
                    tile_scan = tile_scan.cuda()
                res = G_AB(tile_scan.reshape((1,) + tile_scan.shape))  # reshape first for batch axis.
                res_np = res.data.cpu().numpy() if cuda else res.data.numpy()  # get numpy data
                res_np = np.moveaxis(res_np, 1, 3)  # to channels last.
                res_np = (res_np[0] + 1) / 2  # shift pixel values to [0,1] range
                res = numpy2vips(res_np)  # convert to pyvips.Image
                ver_image = res if not ver_image else ver_image.join(res, "vertical")  # "stack" vertically
            image = ver_image if not image else image.join(ver_image, "horizontal")  # "stack" horizontally
        if args.save_linear:
            save(args, i, image, scan)
        else:
            save(args, i, image)


def main_fancy(args, dataset, G_AB, transform, numpy2vips, cuda):
    size = args.patch_size
    for i in range(len(dataset)):
        scan = dataset[i]

        if args.verbose:
            print('Transforming image {} of {}'.format(i + 1, len(dataset)))

        scan = pad_image(scan, size // 2)
        tiles = TileMosaic(scan, (size, size))
        if args.debug:
            x_count = 0
        for x_pos in tqdm.trange(0, scan.width - size - 1, size // 4):
            if args.debug and x_count > 5:
                break
            if args.debug:
                y_count = 0
            for y_pos in range(0, scan.height - size - 1, size // 4):
                if args.debug and y_count > 5:
                    break
                tile_scan = scan.crop(x_pos, y_pos, size, size)  # "grab" square window/patch from image.
                tile_scan = transform(tile_scan)  # convert to torch tensor and channels first.
                if cuda:
                    tile_scan = tile_scan.cuda()
                res = G_AB(tile_scan.unsqueeze(0))  # reshape first for batch axis.
                res_np = res.data.cpu().numpy() if cuda else res.data.numpy()  # get numpy data
                res_np = np.moveaxis(res_np, 1, 3)  # to channels last.
                res_np = (res_np[0] + 1) / 2  # shift pixel values to [0,1] range
                res_vips = numpy2vips(res_np)  # convert to pyvips.Image
                res_vips = res_vips.crop(1, 1, size, size)
                tiles.add_tile(res_vips, x_pos, y_pos)
                y_count += 1
            x_count += 1
        image = tiles.get_mosaic()
        if args.save_linear:
            save(args, i, image, scan)
        else:
            save(args, i, image)
        break


def save(args, i, transformed, linear=None):
    transformed *= 255.0
    output_file = os.path.join(args.output, '{}{}.{}'.format(args.prefix, i, args.format))
    if args.verbose:
        print('Saving transformed image to ' + output_file)
    if args.compression:
        transformed.tiffsave(output_file, tile=True, compression='jpeg', Q=90)
    else:
        transformed.write_to_file(output_file)
    if linear:
        output_file = os.path.join(args.output, '{}_linear_{}.{}'.format(args.prefix, i, args.format))
        if args.verbose:
            print('Saving linear transform image to ' + output_file)
        (linear * 255.0).write_to_file(output_file)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Transform CM whole-slides to H&E.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('directory', type=str, help='directory with mosaic* directories')
    parser.add_argument('--models-dir', required=True, help='directory with saved models')
    parser.add_argument('-o', '--output', required=True, help='output directory')
    parser.add_argument('--prefix', default='scan', help='output files prefix PREFIX')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--format', default='jpg', help='output image format')
    group.add_argument('--compression', action='store_true',
                       help='apply JPEG compression, assumes input images are in TIFF format.')
    parser.add_argument('--epoch', type=int, default=199, help='epoch to get model from.')
    parser.add_argument('--patch-size', type=int, default=2049, help='size in pixels of patch/window.')
    parser.add_argument('--dataset-name', type=str, default='conf_data6', help='name of the saved model dataset.')
    parser.add_argument('--save-linear', action='store_true',
                        help="save linearly stained image (input of model) to '*_linear_*'.")
    parser.add_argument('--overlap', action='store_true',
                        help='overlapping tiles (WSI inference technique by Thomas de Bel et al.)')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    if args.verbose:
        print(args)

    # check if output dir exists
    if not os.path.isdir(args.output):
        if args.verbose:
            print('Output directory does not exist, creating it...')
        os.mkdir(args.output)

    cuda = True if torch.cuda.is_available() else False
    numpy2vips = numpy_pyvips.Numpy2Vips()
    transform = transforms.Compose([numpy_pyvips.Vips2Numpy(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ])
    dataset = SkinDataset(args.directory, stain=True,
                          transform_F=transforms.Lambda(lambda x: x / 65535),
                          transform_R=transforms.Lambda(lambda x: x / 65535))

    G_AB = cyclegan.models.GeneratorResNet(res_blocks=9)
    if cuda:
        G_AB = G_AB.cuda()
    G_AB.load_state_dict(torch.load('%s/%s/G_AB_%d.pth' % (args.models_dir, args.dataset_name, args.epoch)))

    G_AB.eval()
    with torch.no_grad():
        if args.overlap:
            main_fancy(args, dataset, G_AB, transform, numpy2vips, cuda)
        else:
            main(args, dataset, G_AB, transform, numpy2vips, cuda)
