import pyvips
from PIL import Image
import numpy as np


# map np dtypes to vips
dtype_to_format = {
    'uint8': 'uchar',
    'int8': 'char',
    'uint16': 'ushort',
    'int16': 'short',
    'uint32': 'uint',
    'int32': 'int',
    'float32': 'float',
    'float64': 'double',
    'complex64': 'complex',
    'complex128': 'dpcomplex',
}
# map vips formats to np dtypes
format_to_dtype = {
    'uchar': np.uint8,
    'char': np.int8,
    'ushort': np.uint16,
    'short': np.int16,
    'uint': np.uint32,
    'int': np.int32,
    'float': np.float32,
    'double': np.float64,
    'complex': np.complex64,
    'dpcomplex': np.complex128,
}


class Numpy2Vips:
    """numpy array to vips image."""

    def __call__(self, *args, **kwargs):
        return self.numpy2vips(args[0])

    @staticmethod
    def numpy2vips(a):
        height, width, bands = a.shape
        linear = a.reshape(width * height * bands)
        return pyvips.Image.new_from_memory(linear.data, width, height, bands,
                                            dtype_to_format[str(a.dtype)])


class TorchTensor2Vips:
    """torch.Tensor to vips image.

    Assumes input tensor has [Batch,Channel,Height,Width] axis.
    """

    def __init__(self):
        self.numpy2vips = Numpy2Vips()

    def __call__(self, tensor):
        # keep only first element in batch axis
        # and change channel axis to last
        tensor_t = tensor.cpu().detach()[0].permute(1, 2, 0)
        return self.numpy2vips(tensor_t.numpy())


class Vips2Numpy:
    """vips image to numpy array."""

    def __call__(self, *args, **kwargs):
        return self.vips2numpy(args[0])

    @staticmethod
    def vips2numpy(vi):
        return np.ndarray(buffer=vi.write_to_memory(),
                          dtype=format_to_dtype[vi.format],
                          shape=[vi.height, vi.width, vi.bands])


if __name__ == '__main__':
    import sys
    import time

    if len(sys.argv) != 3:
        print('usage: {0} input-filename output-filename'.format(sys.argv[0]))
        sys.exit(-1)

    vips2numpy = Vips2Numpy()
    numpy2vips = Numpy2Vips()

    # load with PIL
    start_pillow = time.time()
    pillow_img = np.asarray(Image.open(sys.argv[1]))
    print('Pillow Time:', time.time() - start_pillow)
    print('pil shape', pillow_img.shape)

    # load with vips to a memory array
    start_vips = time.time()
    img = pyvips.Image.new_from_file(sys.argv[1])
    np_3d = vips2numpy(img)

    print('Vips Time:', time.time() - start_vips)
    print('vips shape', np_3d.shape)

    # make a vips image from the numpy array
    vi = numpy2vips(pillow_img)

    # verify we have the same result
    # this can be non-zero for formats like jpg if the two libraries are using
    # different libjpg versions ... try with png instead
    print('Average pil/vips difference:', (vi - img).avg())

    # and write back to disc for checking
    vi.write_to_file(sys.argv[2])

