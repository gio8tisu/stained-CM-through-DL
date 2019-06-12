import os

from PIL import Image
import pyvips
import numpy as np
import torch
from torchvision import transforms
import tqdm

import cyclegan.models
import numpy_pyvips
from datasets import ScansDataset
from utils import TileMosaic, pad_image

pyvips.cache_set_max(0)

from itertools import product

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
        save(args, i, image, scan)


def main_fancy(args, dataset, G_AB, transform, numpy2vips, cuda):
    size = args.patch_size
    for i in range(len(dataset)):
        scan = dataset[i]

        if args.verbose:
            print('Transforming image {} of {}'.format(i + 1, len(dataset)))
        scan = pad_image(scan, size // 2)
        tiles = TileMosaic(scan, (size, size))
        for x_pos in tqdm.trange(0, scan.width - size - 1, size // 4):
            for y_pos in range(0, scan.height - size - 1, size // 4):
                tile_scan = scan.crop(x_pos, y_pos, size, size)  # "grab" square window/patch from image.
                tile_scan = transform(tile_scan)  # convert to torch tensor and channels first.
                if cuda:
                    tile_scan = tile_scan.cuda()
                res = G_AB(tile_scan.unsqueeze(0))  # reshape first for batch axis.
                res_np = res.data.cpu().numpy() if cuda else res.data.numpy()  # get numpy data
                res_np = np.moveaxis(res_np, 1, 3)  # to channels last.
                res_np = (res_np[0] + 1) / 2  # shift pixel values to [0,1] range
                res_vips = numpy2vips(res_np)  # convert to pyvips.Image
                tiles.add_tile(res_vips)
                break
            break

        image = tiles.build_mosaic()
        save(args, i, image, scan)



def main_fancy_marc(args, dataset, G_AB, transform, numpy2vips, cuda):
    size = args.patch_size

    for i in range(len(dataset)):
        scan = dataset[i]

        (scan*255.0).write_to_file('test2.jpg')

        if args.verbose:
            print('Transforming image {} of {}'.format(i + 1, len(dataset)))
        scan = pad_image_marc(scan, size // 2)
        tiles = TileMosaic(scan, (size, size))

        # 2048, 2048 --> 1024

        x_positions = range(0, scan.width - size - 1, size // 2)
        y_positions = range(0, scan.height - size - 1, size // 2)

        iterator = product(x_positions, y_positions)

        counter_x = 0
        for x_pos in tqdm.trange(0, scan.width - size - 1, size // 2):
            counter_y = 0
            for y_pos in range(0, scan.height - size - 1, size // 2):
                tile_scan = scan.crop(x_pos, y_pos, size, size)  # "grab" square window/patch from image.
                tile_scan = transform(tile_scan)  # convert to torch tensor and channels first.

                if cuda:
                    tile_scan = tile_scan.cuda()

                res = G_AB(tile_scan.unsqueeze(0))  # reshape first for batch axis.
                res_np = res.data.cpu().numpy() if cuda else res.data.numpy()  # get numpy data
                res_np = np.moveaxis(res_np, 1, 3)  # to channels last.
                res_np = (res_np[0] + 1) / 2  # shift pixel values to [0,1] range
                res_vips = numpy2vips(res_np)  # convert to pyvips.Image
                tiles.add_tile(res_vips, (x_pos, y_pos))

                counter_y += 1

            counter_x += 1


        print('Saving')
        image = tiles.build_mosaic()
        (image * 255.0).write_to_file('test.jpg')
        quit()

        save(args, i, image, scan)
        quit()


def save(args, i, image, scan):
    image = image*255.0
    output_file = os.path.join(args.output, '{}{}.{}'.format(args.prefix, i, args.format))
    if args.verbose:
        print('Saving transformed image to ' + output_file)
    if args.compression:
        image.tiffsave(output_file, tile=True, compression='jpeg', Q=90)
    else:
        image.write_to_file(output_file)
    if args.save_linear:
        output_file = os.path.join(args.output, '{}{}.{}'.format(args.save_linear, i, args.format))
        if args.verbose:
            print('Saving linear transform image to ' + output_file)
        scan.write_to_file(output_file)


def pad_image(image, padding):
    """Zero-pad image.

    :param padding: how many pixel to pad by on each side.

    TODO: needs optimization.
    """
    background = numpy_pyvips.Vips2Numpy.vips2numpy(image) * 0
    background.resize((image.height + 2 * padding, image.width + 2 * padding, 3), refcheck=False)
    background = numpy_pyvips.Numpy2Vips.numpy2vips(background)
    return background.insert(image, padding, padding)

def pad_image_marc(image, padding):
    """Zero-pad image.

    :param padding: how many pixel to pad by on each side.
    """
    background = pyvips.Image.black(image.width + 2 * padding, image.height +  2 * padding, bands = image.bands).copy(interpretation='rgb')
    background = background.insert(image, padding, padding)

    return background


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Transform CM whole-slides to H&E.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('directory', type=str, help='directory with mosaic* directories')
    parser.add_argument('-o', '--output', required=True, help='output directory')
    parser.add_argument('--prefix', default='scan', help='output files prefix PREFIX')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--format', default='jpg', help='output image format')
    group.add_argument('--compression', action='store_true',
                       help='apply JPEG compression, assumes input images are in TIFF format.')
    parser.add_argument('--epoch', type=int, default=199, help='epoch to get model from.')
    parser.add_argument('--patch-size', type=int, default=2048, help='size in pixels of patch/window.')
    parser.add_argument('--dataset-name', type=str, default='conf_data6', help='name of the saved model dataset.')
    parser.add_argument('--save-linear', metavar='LIN_PREFIX',
                        help='save linearly stained image (input of model) to LIN_PREFIX.')
    parser.add_argument('--overlap', action='store_true',
                        help='overlapping tiles (WSI inference technique by Thomas de Bel et al.)')
    parser.add_argument('-v', '--verbose', action='store_true', default=True)

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
    dataset = ScansDataset(args.directory, stain=True,
                           transform_F=transforms.Lambda(lambda x: x / 65535),
                           transform_R=transforms.Lambda(lambda x: x / 65535))

    G_AB = cyclegan.models.GeneratorResNet(res_blocks=9)
    if cuda:
        G_AB = G_AB.cuda()
    G_AB.load_state_dict(torch.load('/media/marc/data_disk/confocal/staining/sgarcia/saved_models/%s/G_AB_%d.pth' % (args.dataset_name, args.epoch)))

    G_AB.eval()
    with torch.no_grad():
        if args.overlap:
            main_fancy_marc(args, dataset, G_AB, transform, numpy2vips, cuda)
        else:
            main(args, dataset, G_AB, transform, numpy2vips, cuda)
