"""Image transformation classes for use with pyvips."""

import pyvips

import numpy_pyvips


class VirtualStainer:
    """Class for digitally staining CM to H&E histology using Daniel S. Gareau technique."""

    def __init__(self):
        self.H = [0.30, 0.20, 1]
        self.one_minus_H = list(map(lambda x: 1 - x, self.H))
        self.E = [1, 0.55, 0.88]
        self.one_minus_E = list(map(lambda x: 1 - x, self.E))

    def __call__(self, sample_R, sample_F):
        """Apply staining transformation and return pyvips image.

        :type f_instance: pyvips.Image with range [0,1]
        :type r_instance: pyvips.Image with range [0,1]
        """
        f_res = sample_F * self.one_minus_H
        r_res = sample_R * self.one_minus_E

        image = 1 - f_res - r_res
        return image.copy(interpretation=pyvips.enums.Interpretation.RGB)


class MultiplicativeNoise:
    """Multiply by random variable."""

    def __init__(self, random_variable, **parameters):
        """

        :param random_variable: numpy.random distribution function.
        """
        self.random_variable = random_variable
        self.parameters = parameters
        self.numpy2vips = numpy_pyvips.Numpy2Vips()

    def __call__(self, img: pyvips.Image):
        """return clean image and contaminated image."""
        size = (img.height, img.width, img.bands)
        r = self.numpy2vips(self.random_variable(size=size, **self.parameters))
        return r * img


class CMNormalizer:
    """Normalize CM sample with different methods.

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
            min_R = sample_R.min()
            min_F = sample_F.min()
            max_R = sample_R.max()
            max_F = sample_F.max()
            min_ = min_R if min_R > min_F else min_F
            max_ = max_R if max_R > max_F else max_F
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
        """Normalize pyvips.Image by min and max.

        Intended to be used with torch.transforms.Lambda
        """
        if min_ is None:
            min_ = img.min()
        if max_ is None:
            max_ = img.max()
        return (img - min_) / (min_ - max_)
