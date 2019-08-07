import os
from itertools import product

from PIL import Image
import numpy as np
import pyvips
import torch
from torchvision import transforms
import imageio

import numpy_pyvips
from datasets import SkinCMDataset
from translate_wholeslides import save
from train_despeckling import get_model


def main(args, dataset, model, to_tensor, numpy2vips, cuda):
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
                res = model(tile_scan.unsqueeze(0))
                res_np = res.data.cpu().numpy() if cuda else res.data.numpy()  # get numpy data
                res_np = np.moveaxis(res_np, 1, 3)  # to channels last.
                res_np = res_np[0]
                res = numpy2vips(res_np)  # convert to pyvips.Image
                ver_image = res if not ver_image else ver_image.join(res, "vertical")  # "stack" vertically
            image = ver_image if not image else image.join(ver_image, "horizontal")  # "stack" horizontally
        save(args, i, image)


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
    group.add_argument('--compression', action='store_true',
                       help='apply JPEG compression, assumes input images are in TIFF format.')
    parser.add_argument('--checkpoint-dir', required=True, help='directory with stored model checkpoints.')
    parser.add_argument('--epoch', type=int, help='epoch to get model from. (default: latest)')
    parser.add_argument('--model', default='log_add', help='model name.')
    parser.add_argument('--layers', default=6, type=int, help='number of convolutional layers.')
    parser.add_argument('--filters', default=64, type=int,
                        help='number of filters on each convolutional layer.')
    parser.add_argument('--patch-size', type=int, default=1024, help='size in pixels of patch/window.')
    parser.add_argument('-v', '--verbose', action='store_true')

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
    to_tensor = transforms.Compose([numpy_pyvips.Vips2Numpy(),
                                    transforms.ToTensor()])
    dataset = SkinCMDataset(args.directory, only_R=True,
                            transform_R=transforms.Lambda(lambda x: x / 65535))

    model = get_model(args.model, args.layers, args.filters)
    if cuda:
        model = model.cuda()
    model_path = os.path.join(
        args.checkpoint_dir, '{}_{}.h5'.format(args.model,
                                               'epoch' + str(args.epoch) if args.epoch else 'latest'))
    model.load_state_dict(torch.load(model_path))

    model.eval()
    with torch.no_grad():
        main(args, dataset, model, to_tensor, numpy2vips, cuda)
