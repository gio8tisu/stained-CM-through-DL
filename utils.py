import numpy as np
import pyvips

import numpy_pyvips


pyvips.cache_set_max(0)


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

    def __init__(self, original,
                 tile_shape=(2049, 2049), center_crop=(1025, 1025), window_type=0.25):
        """Constructor.

        :param original: original image, padded by tile_shape / 2.
        :type original: pyvips.Image
        """
        self.tile_shape = tile_shape

        self.background = pyvips.Image.black(original.width, original.height, bands=original.bands).copy(
            interpretation=pyvips.enums.Interpretation.RGB)

        if center_crop[0] > tile_shape[0] or center_crop[1] > tile_shape[1]:
            raise ValueError('center_crop dimensions should be smaller than tile_shape')
        if center_crop is None:
            self.crop = self.tile_shape
        else:
            self.crop = center_crop

        self.result = None

        self.weights = self._define_weight_matrix(window=window_type)

    def _define_weight_matrix(self, window='pyramid', center=None):
        """Return a pyvips.Image with weight values."""
        if window == 'pyramid':
            if center is None:  # use the middle of the image
                center = [self.crop[0] // 2, self.crop[1] // 2]  # center pixel
            x, y = np.meshgrid(np.arange(self.crop[1]), np.arange(self.crop[0]))
            w = 1 - np.maximum(np.abs(x - center[0]) / center[0], np.abs(y - center[1]) / center[1])
            return numpy_pyvips.Numpy2Vips.numpy2vips(w.reshape(w.shape + (1,)))
        elif window == 'circular':
            if center is None:  # use the middle of the image
                center = [self.crop[0] // 2, self.crop[1] // 2]  # center pixel
            Y, X = np.ogrid[:self.crop[1], :self.crop[0]]
            dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
            dist_from_center /= np.max(dist_from_center)
            w = 1 - dist_from_center
            return numpy_pyvips.Numpy2Vips.numpy2vips(w.reshape(w.shape + (1,)))
        elif isinstance(window, (int, float)):
            # multiplying by a scalar is equivalent to multiplying
            # by a constant pyvips.Image.
            return window
        else:
            raise NotImplementedError

    def add_tile(self, tile, x_pos, y_pos):
        """Add tile to build mosaic.

        :type tile: pyvips.Image
        :param x_pos: x position of upper-left corner.
        :param y_pos: y position of upper-left corner.
        """

        if (tile.height, tile.width) != self.tile_shape:
            raise ValueError('Tile is not the correct shape.')
        # crop borders if necessary.
        if self.crop != self.tile_shape:
            tile = tile.crop(tile.width // 2 - self.crop[1] // 2,
                             tile.height // 2 - self.crop[0] // 2,
                             *self.crop)
        tile *= self.weights
        tile_in_bg = self.background.copy().insert(tile, x_pos, y_pos)
        self.result = self.result + tile_in_bg if self.result else tile_in_bg
        # TODO try this:
        # in __init__:
        # self.result = pyvips.Image.black(original.width, original.height, bands=original.bands).copy(
        #    interpretation=pyvips.enums.Interpretation.RGB)
        # in add_tile:
        # self.result.draw_image(tile, x_pos, y_pos, mode=pyvips.enums.Combine.SUM)

    def get_mosaic(self):
        """return mosaic from tiles."""
        return self.result


def pad_image(image, padding):
    """Zero-pad image.

    :param padding: by how many pixels to pad by on each side.
    """
    background = pyvips.Image.black(image.width + 2 * padding,
                                    image.height + 2 * padding,
                                    bands=image.bands)
    return background.insert(image, padding, padding)


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

    tile_shape = (1025, 1025)
    center_crop = (513, 513)
    image_padded = pad_image(image, 512)
    start_time = time.time()
    tile_mosaic = TileMosaic(image_padded, tile_shape, center_crop)

    # transform by tiles and "feed" to TileMosaic object
    for y_pos in range(0, image_padded.height - tile_shape[0] - 1, tile_shape[0] // 4):
        for x_pos in range(0, image_padded.width - tile_shape[1] - 1, tile_shape[1] // 4):
            tile = image_padded.crop(x_pos, y_pos, *tile_shape)  # "grab" square window/patch from image.
            tile_transformed = tile.flip('horizontal')
            tile_mosaic.add_tile(tile_transformed, x_pos, y_pos)
    result = tile_mosaic.get_mosaic()
    end_time = time.time()

    # save image
    result.write_to_file('{}_transformed.{}'.format(*sys.argv[2].split('.')))
    print('Execution time: {}'.format(end_time - start_time))
