import random

import pyvips
import numpy as np
import torch
import torchvision.transforms.functional as TF


class VirtualStainer:
    """Class for digitally staining CM to H&E histology using Daniel S. Gareau technique."""

    def __init__(self):
        self.H = [0.30, 0.20, 1]
        self.one_minus_H = list(map(lambda x: 1 - x, self.H))
        self.E = [1, 0.55, 0.88]
        self.one_minus_E = list(map(lambda x: 1 - x, self.E))

    def __call__(self, sample_R, sample_F):
        """Apply staining transformation and return pyvips image.

        :arg sample_R: pyvips.Image or numpy array with range [0,1]
        :arg sample_F: pyvips.Image or numpy array with range [0,1]
        """
        if (isinstance(sample_F, pyvips.Image)
                and isinstance(sample_R, pyvips.Image)):
            f_res = sample_F * self.one_minus_H
            r_res = sample_R * self.one_minus_E

            image = 1 - f_res - r_res
            return image.copy(interpretation=pyvips.enums.Interpretation.RGB)

        # assumes sample_F and sample_R are numpy arrays
        f_res = sample_F * np.array(self.one_minus_H).reshape((3, 1, 1))
        r_res = sample_R * np.array(self.one_minus_E).reshape((3, 1, 1))

        return 1 - f_res - r_res


class MultiplicativeNoise:
    """Multiply by random variable."""

    def __init__(self, random_variable, **parameters):
        """

        :param random_variable: numpy.random distribution function.
        """
        self.random_variable = random_variable
        self.parameters = parameters

    def __call__(self, img):
        """return clean image and contaminated image."""
        noise = torch.tensor(
            self.random_variable(size=img.size(), **self.parameters),
            device=img.device, dtype=img.dtype, requires_grad=False
        )
        return img * noise, img


class CMMinMaxNormalizer:
    """Min-max normalize CM sample with different methods.

    Independent method "min-max" normalizes each mode separately.
    Global method "min-max" normalizes with global min and max values.
    Average method "min-max" normalizes with min and max values
        of the average image.
    """

    def __init__(self, method):
        assert method in ('independent', 'global', 'average')
        self.method = method

    def __call__(self, sample_R, sample_F):
        if self.method == 'independent':
            new_R = self._normalize(sample_R)
            new_F = self._normalize(sample_F)
        elif self.method == 'global':
            # compute min and max values.
            min_R, max_R = sample_R.min(), sample_R.max()
            min_F, max_F = sample_F.min(), sample_F.max()
            # get global min and max.
            min_ = min_R if min_R > min_F else min_F
            max_ = max_R if max_R > max_F else max_F
            # normalize with global min and max.
            new_R = self._normalize(sample_R, min_, max_)
            new_F = self._normalize(sample_F, min_, max_)
        else:  # self.method == average
            avg = (sample_R + sample_F) / 2
            min_ = avg.min()
            max_ = avg.max()
            new_R = self._normalize(sample_R, min_, max_)
            new_F = self._normalize(sample_F, min_, max_)
        return new_R, new_F

    @staticmethod
    def _normalize(img, min_=None, max_=None):
        """Normalize pyvips.Image by min and max."""
        if min_ is None:
            min_ = img.min()
        if max_ is None:
            max_ = img.max()
        return (img - min_) / (max_ - min_)


class CMRandomCrop:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, R, F):
        r_height, r_width = R.size
        f_height, f_width = F.size
        assert r_height == f_height
        assert r_width == f_width
        rand_i = random.randrange(r_height // 2)
        rand_j = random.randrange(r_width // 2)

        R = TF.crop(R, rand_i, rand_j, self.height, self.width)
        F = TF.crop(F, rand_i, rand_j, self.height, self.width)
        return R, F


class CMRandomHorizontalFlip:
    def __call__(self, R, F):
        if random.random() > 0.5:
            R = TF.hflip(R)
            F = TF.hflip(F)
        return R, F


class CMRandomVerticalFlip:
    def __call__(self, R, F):
        if random.random() > 0.5:
            R = TF.vflip(R)
            F = TF.vflip(F)
        return R, F


class CMToTensor:
    def __call__(self, R, F):
        R = TF.to_tensor(R)
        F = TF.to_tensor(F)
        return torch.cat((R, F))
