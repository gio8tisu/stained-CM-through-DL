import os
from itertools import product

from PIL import Image
import numpy as np
import pyvips
import torch
from torchvision import transforms

import cyclegan.models
import numpy_pyvips


Image.MAX_IMAGE_PIXELS = None

'''
DET#1: F
DET#2: R
'''


class VirtualStainer:
    """Class for digitally staining CM to H&E histology using Daniel S. Gareau technique."""

    def __init__(self):
        self.H = [0.30, 0.20, 1]
        self.one_minus_H = list(map(lambda x: 1 - x, self.H))
        self.E = [1, 0.55, 0.88]
        self.one_minus_E = list(map(lambda x: 1 - x, self.E))

    def __call__(self, *args, **kwargs):
        return self.virtual_staining(args[0], args[1])

    def virtual_staining(self, f_instance, r_instance):
        f_res = f_instance * self.one_minus_H
        r_res = r_instance * self.one_minus_E

        image = 1 - f_res - r_res
        return image.copy(interpretation=pyvips.enums.Interpretation.RGB)


class ScansDataset(torch.utils.data.Dataset):
    """CM scans dataset with possibility to (linearly) stain."""

    def __init__(self, root_dir, stain=True, transform_stained=None, transform_F=None, transform_R=None):
        """
        Args:
            root_dir: str. Directory with "mosaics"
            stain: bool. Stain CM image using VirtualStainer
            scale_by: int. Divide pixel values prior to staining.
            transform_stained: callable object. Apply transform to stained image.
            transform_F: callable object. Apply transform to F-mode image.
            transform_R: callable object. Apply transform to R-mode image.
        """
        self.root_dir = root_dir
        self.transform_stained = transform_stained
        self.transform_F = transform_F
        self.transform_R = transform_R
        self.scans = self._list_scans()
        if stain:
            self.stainer = VirtualStainer()

    def __len__(self):
        return len(self.scans)

    def __getitem__(self, item):
        """Get CM image.
        If stain, return stained image. Return both modes otherwise.
        """
        f_file = self.scans[item] + '/DET#2/highres_raw.tif'
        r_file = self.scans[item] + '/DET#1/highres_raw.tif'
        f_img = pyvips.Image.new_from_file(f_file, access='sequential')
        r_img = pyvips.Image.new_from_file(r_file, access='sequential')

        if self.transform_F:
            f_img = self.transform_F(f_img)
        if self.transform_R:
            r_img = self.transform_R(r_img)
        if self.stainer:
            img = self.stainer(f_img, r_img)
            if self.transform_stained:
                return self.transform_stained(img)
            return img
        return {'F': f_img, 'R': r_img}

    def _list_scans(self):
        scans = []
        for root, dirs, files in os.walk(self.root_dir):
            if 'mosaic' in root.split('/')[-1]:
                scans.append(root)

        scans = sorted(list(set(scans)))
        return scans


def main(args):
    numpy2vips = numpy_pyvips.Numpy2Vips()
    composed_transform = transforms.Compose([numpy_pyvips.Vips2Numpy(scale_by=65536),
                                             transforms.ToTensor(),
                                             ])
    dataset = ScansDataset(args.directory, stain=True, transform_stained=composed_transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                             shuffle=False, num_workers=2)
    G_AB = cyclegan.models.GeneratorResNet(res_blocks=9)
    G_AB.load_state_dict(torch.load('saved_models/%s/G_AB_%d.pth' % (args.dataset_name, args.epoch)))
    for i, scan in enumerate(dataloader):
        # size = 512
        # positions = list(product(range(0, f_img.width - size - 1, size), range(0, f_img.height - size - 1, size)))
        # for position in tqdm.tqdm(positions, total=len(positions)):
        #     tile_f = f_img.crop(position[0], position[1], size, size)
        #     tile_r = r_img.crop(position[0], position[1], size, size)
        #     image = stainer(tile_f / 65536, tile_r / 65536)
        #     image_np = numpy_pyvips.vips2numpy(image)
        #
        #     # transformar...
        #     patch = np.moveaxis(image_np, 3, 1)  # to channels first
        #
        #     # (num_batch, channels, width, height)
        #     patch = torch.Tensor(patch)
        #     patch = torch.autograd.Variable(patch, requires_grad=False)
        #
        #     res_np = G_AB(patch)
        #
        #     res_np = res_np.data.numpy()  # (1, 3, 512, 512)
        #     res_np = np.moveaxis(res_np, 1, 3)  # to channels last
        #     res = numpy_pyvips.numpy2vips(res_np)
        res = G_AB(scan)
        output_file = os.path.join(args.output, '{}#{}.{}'.format(args.prefix, i, args.format))
        numpy2vips(res).write_to_file(output_file)


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
