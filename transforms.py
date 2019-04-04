import pyvips


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
