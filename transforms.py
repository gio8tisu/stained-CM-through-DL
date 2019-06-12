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

    def __call__(self, *args, **kwargs):
        return self.virtual_staining(args[0], args[1])

    def virtual_staining(self, r_instance, f_instance):
        """Apply staining transformation and return pyvips image.

        :type f_instance: pyvips.Image with range [0,1]
        :type r_instance: pyvips.Image with range [0,1]
        """
        f_res = f_instance * self.one_minus_H
        r_res = r_instance * self.one_minus_E

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
