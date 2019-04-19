import os
from itertools import product

from PIL import Image
import numpy as np
import pyvips
import torch
from torchvision import transforms
import imageio

import cyclegan.models
import numpy_pyvips
from datasets import ScansDataset


Image.MAX_IMAGE_PIXELS = None

'''
DET#1: F
DET#2: R
'''


def main(args):
    # check if output dir exists
    if not os.path.isdir(args.output):
        print('Output directory does not exist, creating it...')
        os.mkdir(args.output)
    cuda = True if torch.cuda.is_available() else False
    numpy2vips = numpy_pyvips.Numpy2Vips()
    to_tensor = transforms.Compose([numpy_pyvips.Vips2Numpy(),
                                    transforms.ToTensor()])
    dataset = ScansDataset(args.directory, stain=True,
                           transform_F=transforms.Lambda(lambda x: x / 65535),
                           transform_R=transforms.Lambda(lambda x: x / 65535))
    G_AB = cyclegan.models.GeneratorResNet(res_blocks=9)
    if cuda:
        G_AB = G_AB.cuda()
    G_AB.load_state_dict(torch.load('saved_models/%s/G_AB_%d.pth' % (args.dataset_name, args.epoch)))
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
                tile_scan = to_tensor(tile_scan)  # convert to torch tensor and channels first.
                if cuda:
                    tile_scan = tile_scan.cuda()
                res = G_AB(tile_scan.reshape((1,) + tile_scan.shape))  # reshape first for batch axis.
                res_np = res.data.cpu().numpy() if cuda else res.data.numpy()  # get numpy data
                res_np = np.moveaxis(res_np, 1, 3)  # to channels last.
                res_np = (res_np[0] + 1) / 2  # shift pixel values to [0,1] range
                res = numpy2vips(res_np)  # convert to pyvips.Image
                ver_image = res if not ver_image else ver_image.join(res, "vertical")  # "stack" vertically
            image = ver_image if not image else image.join(ver_image, "horizontal")  # "stack" horizontally

        output_file = os.path.join(args.output, '{}{}.{}'.format(args.prefix, i, args.format))
        if args.verbose:
            print('Saving transformed image to ' + output_file)
        if args.compression:
            image.multiply(2 ** 8 - 1).tiffsave(output_file, tile=True, compression='jpeg', Q=90, bigtiff=True)
        else:
            image.write_to_file(output_file)
        if args.save_linear:
            output_file = os.path.join(args.output, '{}{}.{}'.format(args.save_linear, i, args.format))
            if args.verbose:
                print('Saving linear transform image to ' + output_file)
            if args.compression:
                scan.multiply(2 ** 8 - 1).tiffsave(output_file, tile=True, compression='jpeg', Q=90, bigtiff=True)
            else:
                scan.write_to_file(output_file)


if __name__ == '__main__':
    import argparse
    import tqdm

    parser = argparse.ArgumentParser(description='Transform CM whole-slides to H&E.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('directory', type=str, help='directory with mosaic* directories')
    parser.add_argument('-o', '--output', required=True, help='output directory')
    parser.add_argument('--prefix', default='scan', help='output files prefix PREFIX')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--format', default='tif', help='output image format')
    group.add_argument('--compression', action='store_true')
    parser.add_argument('--epoch', type=int, default=199, help='epoch to get model from.')
    parser.add_argument('--patch-size', type=int, default=512, help='size in pixels of patch/window.')
    parser.add_argument('--dataset-name', type=str, default='conf_data6', help='name of the saved model dataset.')
    parser.add_argument('--save-linear', metavar='LIN_PREFIX',
                        help='save linearly stained image (input of model) to LIN_PREFIX.')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    if args.verbose:
        print(args)
    main(args)
