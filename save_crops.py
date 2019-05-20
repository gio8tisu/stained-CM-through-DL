import os
from itertools import product
import argparse
import tqdm

from datasets import ScansDataset


'''
DET#1: F
DET#2: R
'''


def main():
    size = args.patch_size
    dataset = ScansDataset(args.input_directory)
    for i in range(len(dataset)):
        scan = dataset[i]  # dictionary with keys 'F' and 'R'
        if args.verbose:
            print('Cropping image {} of {}'.format(i + 1, len(dataset)))
        y_x = product(range(0, scan['F'].height - size - 1, args.step),
                      range(0, scan['F'].width - size - 1, args.step))
        for y_pos, x_pos in tqdm.tqdm(y_x):
            for mode, s in scan.items():
                tile_scan = s.crop(x_pos, y_pos, size, size)  # "grab" square window/patch from image.
                save(tile_scan, i, x_pos, y_pos, mode)
    print('Done.')


def save(image, i, x, y, mode):
    file_name = os.path.join(args.output_directory, f'{args.prefix}{i}_{y}-{x}_{mode}.{args.format}')
    if args.verbose:
        print('Saving crop to ' + file_name, end='\r')
    if args.compression:
        image.tiffsave(file_name, tile=True, compression='jpeg', Q=95)
    else:
        image.write_to_file(file_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Save crops from CM whole-slides.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_directory', type=str, help='directory with mosaic* directories')
    parser.add_argument('output_directory', help='output directory')
    parser.add_argument('--prefix', default='', help='output files prefix PREFIX')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--format', default='tif', help='output image format')
    group.add_argument('--compression', action='store_true', help='apply JPEG compression')
    parser.add_argument('--patch-size', type=int, default=512, help='size in pixels of square patch/window')
    parser.add_argument('--step', type=int, help='window step in pixels')
    parser.add_argument('--discard', action='store_true', help='discard patches with mean<2000')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    if args.verbose:
        print(args)

    # check if output dir exists
    if not os.path.isdir(args.output_directory):
        if args.verbose:
            print('Output directory does not exist, creating it...')
        os.makedirs(args.output_directory)

    if not args.step:
        args.step = args.patch_size

    main()
