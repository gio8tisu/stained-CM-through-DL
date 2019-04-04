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
    numpy2vips = numpy_pyvips.Numpy2Vips()
    composed_transform = transforms.Compose([numpy_pyvips.Vips2Numpy(scale_by=65536),
                                             transforms.ToTensor(),
                                             ])
    dataset = ScansDataset(args.directory, stain=True)
    G_AB = cyclegan.models.GeneratorResNet(res_blocks=9)
    G_AB.load_state_dict(torch.load('saved_models/%s/G_AB_%d.pth' % (args.dataset_name, args.epoch)))
    size = 512
    SAVE_I, SAVE_J = 11264, 3072
    for i in range(len(dataset)):
        scan = dataset[i]
        positions = list(product(range(0, scan.width - size - 1, size), range(0, scan.height - size - 1, size)))
        for position in tqdm.tqdm(positions, total=len(positions)):
            tile_scan = scan.crop(position[0], position[1], size, size)
            tile_scan = composed_transform(tile_scan)
            # guardar tile_scan
            if position[0] == SAVE_I and position[1] == SAVE_J:
                imageio.imwrite('tile_scan.png', np.moveaxis(tile_scan.numpy(), 0, 2))
                print('IMAGE TILE SAVED.')
            res = G_AB(tile_scan.reshape((1,) + tile_scan.shape))  # reshape first for batch axis

            res_np = res.data.numpy()  # (1, 3, 512, 512)
            res_np = np.moveaxis(res_np, 1, 3)  # to channels last
            res_np = res_np[0]
            # guardar res_np
            if position[0] == SAVE_I and position[1] == SAVE_J:
                imageio.imwrite('res_np.png', res_np)
                print('IMAGE TILE TRANSFORMED SAVED.')
            res = numpy2vips(res_np) * 65536
            output_file = os.path.join(args.output, '{}_{}_{}-{}.{}'.format(args.prefix, i, position[0],
                                                                            position[1], args.format))
            res.write_to_file(output_file)


if __name__ == '__main__':
    import argparse
    import tqdm

    parser = argparse.ArgumentParser(description='Transform CM whole-slide to H&E.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('directory', type=str, help='name of the dataset.')
    parser.add_argument('-o', '--output', required=True, help='output directory')
    parser.add_argument('--prefix', default='scan', help='output files prefix PREFIX')
    parser.add_argument('--format', default='tif', help='output image format')
    parser.add_argument('--epoch', type=int, default=199, help='epoch to get model from.')
    parser.add_argument('--dataset_name', type=str, default='conf_data6', help='name of the saved model dataset.')

    args = parser.parse_args()
    main(args)
