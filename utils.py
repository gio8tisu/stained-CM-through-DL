import os
import tempfile

import numpy as np
import pyvips

import numpy_pyvips


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
                 tile_shape=(2048, 2048), center_crop=None, window_type=0.25):
        """Constructor.

        :param original: original image, padded by tile_shape / 2.
        :type original: pyvips.Image
        """
        if isinstance(tile_shape, int):
            self.tile_shape = (tile_shape,) * 2
        else:
            self.tile_shape = tile_shape

        self.background = pyvips.Image.black(original.width, original.height, bands=original.bands).copy(
            interpretation=pyvips.enums.Interpretation.RGB)

        if center_crop is None:
            self.crop = self.tile_shape
        elif isinstance(center_crop, int):
            self.crop = (center_crop,) * 2
        else:
            self.crop = center_crop
        if self.crop[0] > self.tile_shape[0] or self.crop[1] > self.tile_shape[1]:
            raise ValueError('center_crop dimensions should be smaller than tile_shape')

        self.weights = self._define_weight_matrix(window=window_type)

        self.file_names = []
        self.tmp_dir = tempfile.TemporaryDirectory(suffix='translate_wholeslide', dir=os.getenv('TMPDIR', None))

    def _define_weight_matrix(self, window='pyramid', center=None):
        """Return a pyvips.Image with weight values."""
        if isinstance(window, (int, float)):
            # multiplying by a scalar is equivalent to multiplying
            # by a constant pyvips.Image.
            return window
        if center is None:  # use the middle of the image
            center = [(self.crop[0] - 1) / 2, (self.crop[1] - 1) / 2]  # center pixel
        Y, X = np.ogrid[:self.crop[1], :self.crop[0]]
        if window == 'pyramid':
            # center and scale.
            X = (X - center[0]) / center[0]
            Y = (Y - center[1]) / center[1]
            w = (1 - np.abs(X)) * (1 - np.abs(Y))
            return numpy_pyvips.Numpy2Vips.numpy2vips(w.reshape(w.shape + (1,)))
        elif window == 'circular':
            dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
            dist_from_center /= np.max(dist_from_center)
            w = 1 - dist_from_center
            return numpy_pyvips.Numpy2Vips.numpy2vips(w.reshape(w.shape + (1,)))
        else:
            raise NotImplementedError("'window' parameter should be a scalar, "
                                      "'pyramid' or 'circular")

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
        self.file_names.append(os.path.join(self.tmp_dir.name, f'{x_pos}-{y_pos}.v'))
        # self.file_names.append(os.path.join(self.tmp_dir, f'{x_pos}-{y_pos}.v'))
        tile.write_to_file(self.file_names[-1])
        # TODO try this:
        # in __init__:
        # self.result = pyvips.Image.black(original.width, original.height, bands=original.bands)
        #                           .copy(interpretation=pyvips.enums.Interpretation.RGB)
        # in add_tile:
        # self.result.draw_image(tile, x_pos, y_pos, mode=pyvips.enums.Combine.SUM)

    def get_mosaic(self):
        """return mosaic from tiles."""
        result = self.background.copy()
        for file in self.file_names:
            crop = pyvips.Image.new_from_file(file)
            # remove extension and split.
            x, y = os.path.basename(file[:-2]).split('-')
            result += self.background.insert(crop, int(x), int(y))

        return result


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

    tile_shape = (2**3 + 1, 2**3 + 1)
    center_crop = (2**2 + 1, 2**2 + 1)
    image_padded = pad_image(image, 2**3)
    start_time = time.time()
    tile_mosaic = TileMosaic(image_padded, tile_shape, center_crop, window_type='pyramid')

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
