import os
import argparse

from torch.utils.data import random_split
from torchvision import transforms

import datasets


def main(dataset):
    train_size = int(args.split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    for i in range(len(train_dataset)):
        scan = train_dataset[i]  # stained sample
        save(scan, i, 'train')
    for i in range(len(test_dataset)):
        scan = test_dataset[i]  # stained sample
        save(scan, i, 'test')
    print('Done.')


def save(image, i, split):
    image *= 255
    file_name = os.path.join(args.output_directory, split, f'{args.prefix}{i}.{args.format}')
    if args.verbose:
        print('Saving crop to ' + file_name, end='\r')
    if args.compression:
        image.tiffsave(file_name, tile=True, compression='jpeg', Q=95)
    else:
        image.write_to_file(file_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create dataset of linearly stained scan crops',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_directory', type=str, help='data root')
    parser.add_argument('output_directory', help='output directory')
    parser.add_argument('--split', type=float, default=0.8, help='train split relative size')
    parser.add_argument('--prefix', default='', help='output files prefix PREFIX')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--format', default='tif', help='output image format')
    group.add_argument('--compression', action='store_true', help='apply JPEG compression')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    if args.verbose:
        print(args)

    # check if output dir exists
    if not os.path.isdir(args.output_directory):
        if args.verbose:
            print('Output directory does not exist, creating it...')
        os.makedirs(args.output_directory + '/train')
        os.mkdir(args.output_directory + '/test')

    dataset = datasets.CMCropsDataset(args.input_directory,
                                      transform_F=transforms.Lambda(lambda x: x / 255),
                                      transform_R=transforms.Lambda(lambda x: x / 255),
                                      stain=True)
    main(dataset)
