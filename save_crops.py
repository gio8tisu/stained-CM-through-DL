import os
from itertools import product
import argparse
import tqdm

import datasets


def main(dataset):
    size = args.patch_size
    for i in range(len(dataset)):
        scan = dataset[i]  # dictionary with keys 'F' and 'R'
        if args.verbose:
            print('Cropping image {} of {}'.format(i + 1, len(dataset)))
        with tqdm.tqdm() as pbar:
            y_pos = 0
            while y_pos < scan['F'].height - size - 1:
                x_pos = 0
                while x_pos < scan['F'].width - size - 1:
                    for mode, s in scan.items():
                        tile_scan = s.crop(x_pos, y_pos, size, size)  # "grab" square window/tile from image.
                        if args.discard:
                            avg = tile_scan.avg()
                            if avg < 3000:
                                if args.verbose:
                                    print(f'Discarding crop {y_pos}-{x_pos} with mean pixel value of {avg}.')
                                break
                        save(tile_scan, i, x_pos, y_pos, mode, args.prefix, args.format)
                    x_pos += args.step
                    pbar.update()
                y_pos += args.step
    print('Done.')


def save(image, i, x, y, mode, prefix='', format_='tiff'):
    file_name = os.path.join(args.output_directory, f'{prefix}{i}_{y}-{x}_{mode}.{format_}')
    if args.verbose:
        print('Saving crop to ' + file_name, end='\r')
    if args.compression:
        image.tiffsave(file_name, compression='jpeg', Q=95, tile=True, pyramid=True)
    else:
        image.write_to_file(file_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Save crops from CM whole-slides.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_directory', type=str, help='data root')
    parser.add_argument('output_directory', help='output directory')
    parser.add_argument('--dataset', choices=['skin', 'colon-CM', 'colon-HE'],
                        help='dataset type. If dataset is colon-HE, borders will be discarded')
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

    if args.dataset == 'skin':
        dataset = datasets.SkinCMDataset(args.input_directory)
    elif args.dataset == 'colon-CM':
        dataset = datasets.ColonCMDataset(args.input_directory)
    elif args.dataset == 'colon-HE':
        dataset = datasets.ColonHEDataset(args.input_directory)
    main(dataset)
