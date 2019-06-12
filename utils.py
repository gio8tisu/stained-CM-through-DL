from math import ceil

import numpy as np
import pyvips

import numpy_pyvips
import cv2


def create_circular_mask(h, w, center=None, radius=None):
    radius = 2*h/3
    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    dist_from_center /= np.max(dist_from_center)
    dist_from_center = 1-dist_from_center
    dist_from_center += np.min(dist_from_center)
    return dist_from_center

class TileMosaic:
    """Class for WSI inference technique by Thomas de Bel et al.

    Attributes
    ----------
        tile_shape (Tuple[int]) - tile shape in pixels (height, width).
        step (Tuple[int]) - step in pixels (height, width).
        crop (Tuple[int]) - pixels to be kept of tile center (height, width).
        tiles (List[pyvips.Image]) - list with tiles (first index corresponds to row).
        num_h (int) - number of "vertical" tiles.
        num_w (int) - number of "horizontal" tiles.

    """

    def __init__(self, original, tile_shape=(2048, 2048)):
        """Constructor.

        :param original: original image, padded by tile_shape / 2.
        :type original: pyvips.Image
        """
        self.tile_shape = tile_shape
        self.crop = (2048, 2048)
        self.steps = (range(0, original.height - tile_shape[0] - 1, tile_shape[0] // 4),
                      range(0, original.width - tile_shape[1] - 1, tile_shape[1] // 4))
        self.tiles = []
        self.background = pyvips.Image.black(original.width, original.height, bands=original.bands).copy(
            interpretation='rgb')

        # weight matrix definition
        x, y = np.meshgrid(np.arange(self.tile_shape[1]), np.arange(self.tile_shape[0]))
        x_cp = self.tile_shape[1] // 2
        y_cp = self.tile_shape[0] // 2
        w = 1 - np.maximum(np.abs(x - x_cp) / x_cp, np.abs(y - y_cp) / y_cp)

        w = create_circular_mask(self.tile_shape[1], self.tile_shape[0])
        w /= 2

        self.weights = numpy_pyvips.Numpy2Vips.numpy2vips(w.reshape(w.shape + (1,)))

    def add_tile(self, tile, pos):
        """Add tile to build mosaic.
        This assumes tiles are feed left-to-right top-to-bottom order and the
        image is padded by tile_shape / 4.

        :type tile: pyvips.Image
        """

        '''
        if (tile.height, tile.width) != self.tile_shape:
            raise ValueError('Tile is not the correct shape.')
        # crop borders

        center = tile.crop(tile.width // 2 - self.crop[1] // 2,
                                   tile.height // 2 - self.crop[0] // 2,
                                   *self.crop)
        '''

        # tile *= self.weights
        tile /= 4
        aux = pyvips.Image.black(self.background.width, self.background.height, bands=self.background.bands).copy(
            interpretation='rgb')
        aux = aux.insert(tile, pos[0], pos[1])

        self.background += aux


    def build_mosaic(self):
        return self.background



if __name__ == '__main__':
    import sys
    import time

    try:
        if sys.argv[1] == 'tile':
            image = pyvips.Image.new_from_file(sys.argv[2])
        else:
            raise Exception
    except:
        print('usage: "{} tile <IMAGE_PATH>" to test TileMosaic class'.format(sys.argv[0]))
        sys.exit(1)

    tile_shape = (512, 512)
    background = numpy_pyvips.Vips2Numpy.vips2numpy(image) * 0
    background.resize((image.height + tile_shape[0], image.width + tile_shape[1], 3), refcheck=False)
    background = numpy_pyvips.Numpy2Vips.numpy2vips(background)
    image_padded = background.insert(image, tile_shape[1] // 2, tile_shape[0] // 2)
    start_time = time.time()
    tile_mosaic = TileMosaic(image_padded, tile_shape)

    # transform by tiles and "feed" to TileMosaic object
    for y_pos in range(0, image_padded.height - tile_shape[0] - 1, tile_shape[0] // 4):
        for x_pos in range(0, image_padded.width - tile_shape[1] - 1, tile_shape[1] // 4):
            tile = image_padded.crop(x_pos, y_pos, *tile_shape)  # "grab" square window/patch from image.
            tile_transformed = tile.invert()
            tile_mosaic.add_tile(tile_transformed)
    result = tile_mosaic.build_mosaic()
    end_time = time.time()

    # save image
    result.write_to_file('{}_transformed.{}'.format(*sys.argv[2].split('.')))
    print('Execution time: {}'.format(end_time - start_time))
