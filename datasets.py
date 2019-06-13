import os
import pathlib
import random

import torch.utils.data
from torchvision import transforms
import pyvips

import numpy_pyvips
from transforms import VirtualStainer, MultiplicativeNoise


'''
DET#1: R
DET#2: F
'''


class ScansDataset(torch.utils.data.Dataset):
    """CM scans dataset with possibility to (linearly) stain."""

    def __init__(self, root_dir, only_R=False, only_F=False, stain=False,
                 transform_stained=None, transform_F=None, transform_R=None):
        """
        Args:
            root_dir (str): Directory with "mosaic" directories.
            only_R (bool): return only R mode.
            only_F (bool): return only F mode. If both only_R and only_F are True,
                           the former takes precedence.
            stain (bool): Stain CM image using VirtualStainer
            transform_stained (callable): Apply transform to stained image.
            transform_F (callable): Apply transform to F-mode image.
            transform_R (callable): Apply transform to R-mode image.
        """
        self.root_dir = root_dir
        if only_R and only_F:
            raise ValueError("Only one (if any) of 'only' options must be true.")
        self.only_R, self.only_F = only_R, only_F
        self.transform_stained = transform_stained
        self.transform_F = transform_F
        self.transform_R = transform_R
        self.scans = self._list_scans()
        self.stainer = VirtualStainer() if stain else None

    def __len__(self):
        return len(self.scans)

    def __getitem__(self, item):
        """Get CM image.
        If stain, return stained image (using transforms.VirtualStainer).
        Return both modes otherwise.
        """
        # load R mode if needed
        if not self.only_F:
            r_img = self.get_r(item)
        # load F mode if needed
        if not self.only_R:
            f_img = self.get_f(item)

        if self.transform_F:
            f_img = self.transform_F(f_img)
        if self.transform_R:
            r_img = self.transform_R(r_img)

        if self.stainer:
            img = self.stainer(f_img, r_img)
            if self.transform_stained:
                return self.transform_stained(img)
            return img

        if self.only_R:
            return r_img
        elif self.only_F:
            return f_img
        return {'F': f_img, 'R': r_img}

    def get_f(self, item):
        f_file = self.scans[item] + '/DET#2/highres_raw.tif'
        f_img = pyvips.Image.new_from_file(f_file)
        return f_img

    def get_r(self, item):
        r_file = self.scans[item] + '/DET#1/highres_raw.tif'
        r_img = pyvips.Image.new_from_file(r_file)
        return r_img

    def _list_scans(self):
        scans = []
        for root, dirs, files in os.walk(self.root_dir):
            if 'mosaic' in root.split('/')[-1]:
                scans.append(root)

        scans = sorted(list(set(scans)))
        return scans


class ScansCropsDataset(ScansDataset):
    """CM scans crops dataset with possibility to (linearly) stain.

    TODO: migrate to PIL.
    """

    def get_f(self, item):
        f_file = self.scans[item][1]  # second element of tuple is F mode
        f_img = pyvips.Image.new_from_file(str(f_file))
        return f_img

    def get_r(self, item):
        r_file = self.scans[item][0]  # first element of tuple is R mode
        r_img = pyvips.Image.new_from_file(str(r_file))
        return r_img

    def _list_scans(self):
        crops_R = self.root_dir.glob('*R.tif')
        crops_F = self.root_dir.glob('*F.tif')
        assert len(crops_F) == len(crops_R)
        crops = list(zip(sorted(crops_R), sorted(crops_F)))

        return crops


class NoisyScansDataset(ScansCropsDataset):
    """Dataset with 512x512 CM crops with speckle noise.

    TODO: migrate to PIL.
    """

    def __init__(self, root_dir, which, noise_args, apply_random_crop=False):
        """

        :param root_dir: Directory with "mosaic" directories.
        :param which: which mode to work with (F or R).
        :param noise_args: dictionary with 'random_variable' and parameter keys.
        :param apply_random_crop: apply 256x256 random crop.
        """
        if which == 'F':
            super(NoisyScansDataset, self).__init__(
                root_dir, only_F=True, transform_F=transforms.Lambda(lambda x: x / 65535))
        elif which == 'R':
            super(NoisyScansDataset, self).__init__(
                root_dir, only_R=True, transform_R=transforms.Lambda(lambda x: x / 65535))
        else:
            raise ValueError("'which' parameter should be 'F' or 'R'")
        self.add_noise = MultiplicativeNoise(**noise_args)
        self.to_tensor = transforms.Compose([numpy_pyvips.Vips2Numpy(), transforms.ToTensor()])
        self.apply_random_crop = apply_random_crop

    def __getitem__(self, item):
        clean = super(NoisyScansDataset, self).__getitem__(item)
        noisy = self.add_noise(clean)
        if self.apply_random_crop:
            x = random.randint(0, 255)
            y = random.randint(0, 255)
            clean = clean.crop(x, y, 256, 256)
            noisy = noisy.crop(x, y, 256, 256)
        return self.to_tensor(noisy), self.to_tensor(clean)
