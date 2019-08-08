import os
import pathlib
import random
from abc import ABCMeta, abstractmethod
import warnings

import torch.utils.data
from torchvision import transforms
import pyvips
from PIL import Image
# import openslide

import numpy_pyvips
from transforms import VirtualStainer, MultiplicativeNoise


def return_prefix_decorator(getitem):
    """Used to wrap __getitem__ method.

    If return_prefix attribute is True, it will make __getitem__ return
    the desired item along with its file prefix using _get_prefix method
    (this can be used to identify the samples).
    """
    def getitem_wrapper(self, item):
        sample = getitem(self, item)
        if self.return_prefix:
            prefix = self.get_prefix(item)
            return sample, prefix
        else:
            return sample
    return getitem_wrapper


class CMDataset(torch.utils.data.Dataset, metaclass=ABCMeta):
    """CM scans dataset abstract class with possibility to (linearly) stain."""
    def __init__(self, transform=None, only_R=False, only_F=False, stain=False,
                 transform_stained=None, transform_F=None, transform_R=None,
                 return_prefix=False):
        """
        Args:
            only_R (bool): return only R mode.
            only_F (bool): return only F mode. If both only_R and only_F are True,
                           the former takes precedence.
            stain (bool): Stain CM image using VirtualStainer.
            transform_stained (callable): Apply transform to stained image.
            transform_F (callable): Apply transform to F-mode image.
            transform_R (callable): Apply transform to R-mode image.
            transform (callable): Apply transform to both modes (after respective transforms).
                                  R and F modes will be used as argument in that order.
        """
        if only_R and only_F:
            raise ValueError("Only one (if any) of 'only' options must be true.")
        self.only_R, self.only_F = only_R, only_F
        self.transform_stained = transform_stained
        self.transform_F = transform_F
        self.transform_R = transform_R
        self.transform = transform
        self.scans = self._list_scans()
        self.stainer = VirtualStainer() if stain else None
        self.return_prefix = return_prefix

    def __len__(self):
        return len(self.scans)

    @abstractmethod
    def get_f(self, item):
        """Return item-th sample F mode."""
        pass

    @abstractmethod
    def get_r(self, item):
        """Return item-th sample R mode."""
        pass

    @abstractmethod
    def get_prefix(self, item):
        """Return item-th sample prefix."""
        pass

    @abstractmethod
    def _list_scans(self):
        pass

    @return_prefix_decorator
    def __getitem__(self, item):
        """Get CM image.

        If stain, return stained image (using transforms.VirtualStainer).
        Return both modes otherwise.
        If return_prefix, return (sample, prefix) tuple.
        """
        # load R mode if needed
        r_img = None if self.only_F else self.get_r(item)
        # load F mode if needed
        f_img = None if self.only_R else self.get_f(item)

        if self.transform_F:
            f_img = self.transform_F(f_img)
        if self.transform_R:
            r_img = self.transform_R(r_img)

        if self.transform:
            r_img, f_img = self.transform(r_img, f_img)

        if self.stainer:
            img = self.stainer(r_img, f_img)
            if self.transform_stained:
                return self.transform_stained(img)
            return img

        if self.only_R:
            return r_img
        elif self.only_F:
            return f_img
        return {'F': f_img, 'R': r_img}


class ColonCMDataset(CMDataset):
    """CM colon scans dataset with possibility to (linearly) stain.

    785: R
    488: F
    """

    def __init__(self, root_dir, **kwargs):
        self.root_dir = pathlib.Path(root_dir)
        super().__init__(**kwargs)

    def _list_scans(self):
        scans_R = list(self.root_dir.glob('**/785.png'))
        scans_F = list(self.root_dir.glob('**/488.png'))
        assert len(scans_F) == len(scans_R)
        # list of (R,F) pairs, needs to be list so it has len().
        scans = list(zip(sorted(scans_R), sorted(scans_F)))

        return scans

    def get_f(self, item):
        f_file = self.scans[item][1]  # second element of tuple is F mode
        f_img = pyvips.Image.new_from_file(str(f_file))
        return f_img

    def get_r(self, item):
        r_file = self.scans[item][0]  # first element of tuple is R mode
        r_img = pyvips.Image.new_from_file(str(r_file))
        return r_img

    def get_prefix(self, item):
        return self.scans[item][0][:-8]


class ColonHEDataset(torch.utils.data.Dataset):
    """H&E colon scans dataset."""

    def __init__(self, root_dir, transform=None, alpha=False):
        """
        Args:
            root_dir (str): Directory with "mosaic" directories.
            transform (callable): Apply transform to image.
            alpha (bool): return slide with alpha channel.
        """
        self.root_dir = pathlib.Path(root_dir)
        self.transform = transform
        self.alpha = alpha
        self.scans = sorted(list(self.root_dir.glob('*.bif')))

    def __len__(self):
        return len(self.scans)

    def __getitem__(self, item):
        """Get max resolution H&E image.

        :return openslide.OpenSlide object
        """
        scan = openslide.OpenSlide(str(self.scans[item]))
        # TODO: transform to PIL
        if self.alpha:
            return scan
        # TODO: remove alpha channel
        # https://stackoverflow.com/questions/9166400/convert-rgba-png-to-rgb-with-pil
        if self.transform:
            scan = self.transform(scan)
        return scan


class SkinCMDataset(CMDataset):
    """CM skin scans dataset with possibility to (linearly) stain.

    DET#1: R
    DET#2: F
    """

    def __init__(self, root_dir, **kwargs):
        self.root_dir = root_dir
        super().__init__(**kwargs)

    def _list_scans(self):
        scans = []
        for root, dirs, files in os.walk(self.root_dir):
            if 'mosaic' in root.split('/')[-1]:
                scans.append(root)

        scans = sorted(list(set(scans)))
        return scans

    def get_f(self, item):
        f_file = self.scans[item] + '/DET#2/highres_raw.tif'
        f_img = pyvips.Image.new_from_file(f_file, access='random')
        return f_img

    def get_r(self, item):
        r_file = self.scans[item] + '/DET#1/highres_raw.tif'
        r_img = pyvips.Image.new_from_file(r_file, access='random')
        return r_img

    def get_prefix(self, item):
        return self.scans[item]


class CMCropsDataset(CMDataset):
    """CM scans crops dataset with possibility to (linearly) stain.

    To extract crops from wholeslides use save_crops.py script.
    """

    def __init__(self, root_dir, **kwargs):
        self.root_dir = pathlib.Path(root_dir)
        super().__init__(**kwargs)

    def _list_scans(self):
        crops_R = {str(r)[:-6] for r in self.root_dir.glob('*R.tif')}
        crops_F = {str(f)[:-6] for f in self.root_dir.glob('*F.tif')}
        if self.only_R:
            crops = crops_R
        elif self.only_F:
            crops = crops_F
        else:
            # if use both modes, use only the crops with both modes available.
            if len(crops_F) != len(crops_R):
                warnings.warn('Number of crops for R and F modes are different. '
                              'Dataset will be only composed by the images with'
                              'both modes available.')
            crops = crops_R & crops_F  # set intersection.
        return sorted(crops)

    def get_f(self, item):
        f_file = self.scans[item] + '_F.tif'
        f_img = Image.open(f_file)
        return f_img

    def get_r(self, item):
        r_file = self.scans[item] + '_R.tif'
        r_img = Image.open(r_file)
        return r_img

    def get_prefix(self, item):
        return os.path.basename(self.scans[item])


class NoisyCMCropsDataset(CMCropsDataset):
    """Dataset with 512x512 CM crops with speckle noise."""

    def __init__(self, root_dir, mode, noise_args, transform=None,
                 return_prefix=False):
        """

        :param root_dir: Directory with "mosaic" directories.
        :param mode: which mode to work with (F or R).
        :param noise_args: dictionary with 'random_variable' and parameter keys.
        """
        if mode == 'F':
            super().__init__(
                root_dir, only_F=True, transform_F=transform,
                return_prefix=return_prefix)
        elif mode == 'R':
            super().__init__(
                root_dir, only_R=True, transform_R=transform,
                return_prefix=return_prefix)
        else:
            raise ValueError("'mode' parameter should be 'F' or 'R'")
        self.add_noise = MultiplicativeNoise(**noise_args)

    def __getitem__(self, item):
        """Return (noisy, clean) tuple.

        If return_prefix, return ((noisy, clean), prefix) tuple.
        Return (noisy, clean) otherwise.
        """
        sample = super().__getitem__(item)
        if self.return_prefix:
            clean, prefix = sample
        noisy = self.add_noise(clean)
        if self.return_prefix:
            return (noisy, clean), prefix
        return noisy, clean

