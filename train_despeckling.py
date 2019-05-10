import argparse
import os.path

import torch
from torch.utils.data import DataLoader, random_split
from torch.nn import MSELoss, L1Loss
from torch.optim import Adam
import numpy as np
from skimage.measure import compare_ssim as ssim
import tqdm

from datasets import NoisyScansDataset
from despeckling import models


def main(args):
    cuda = True if torch.cuda.is_available() else False

    # Define dataset.
    if args.noise == 'gaussian':
        noise_args = {'random_variable': np.random.normal,
                      'loc': 1, 'scale': 0.2}
    elif args.noise == 'gamma':
        noise_args = {'random_variable': np.random.gamma,
                      'shape': 1, 'scale': 1}
    dataset = NoisyScansDataset(args.data_root, 'F', noise_args, apply_random_crop=True)  # returns (noisy, clean) tuple
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4)

    # Define model and loss criterion.
    if args.model == 'log_add':
        model = models.LogSubtractDespeckle()
    else:
        raise NotImplementedError(args.model + 'model does not exist.')
    if args.criterion == 'mse':
        criterion = MSELoss()
    elif args.criterion == 'l1':
        criterion = L1Loss()
    else:
        raise NotImplementedError(args.criterion + 'loss criterion is not supported.')
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    # Define optimizer.
    if args.optim == 'adam':
        optimizer = Adam(params=model.parameters(), lr=args.lr)
    else:
        raise NotImplementedError(args.optim + 'optimizer is not supported.')

    # Training process.
    for epoch in range(args.epochs):
        model.train()
        med_loss = 0
        input_and_target = enumerate(train_dataloader)
        if args.verbose:
            print('Epoch {} of {}'.format(epoch, args.epochs))
            input_and_target = tqdm.tqdm(input_and_target, total=len(train_dataloader))
        for i, (x, target) in input_and_target:
            x, target = x.float().cuda(), target.float().cuda()

            optimizer.zero_grad()
            output = model(x)

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            med_loss += loss.data.cpu().numpy()

            if args.verbose:
                input_and_target.set_description('Loss = ' + str("{0:.3f}".format(med_loss/(i+1))))

        model.eval()
        med_loss = 0
        prev_loss = 0
        mean_ssim1 = 0
        mean_ssim2 = 0

        input_and_target = enumerate(val_dataloader)
        if args.verbose:
            input_and_target = tqdm.tqdm(input_and_target, total=len(val_dataloader))
        for i, (x, target) in input_and_target:
            x, target = x.cuda(), target.cuda()
            output = model(x)

            output = output.data.cpu().numpy()[0, 0]
            aux_img = x.data.cpu().numpy()[0, 0]

            # output *= (np.max(aux_img) - np.min(aux_img)) / (np.max(output) - np.min(output))

            output += 1
            output /= 2
            output *= 255
            output = output.astype(np.uint8)

            output = np.median(output)
            output = (output / 255.0 - 0.5) * 2
            output = torch.tensor([[output]], dtype=torch.float).cuda()

            mean_ssim1 += ssim(output.data.cpu().numpy()[0, 0], target.data.cpu().numpy()[0, 0],
                               data_range=2)

            mean_ssim2 += ssim(x.data.cpu().numpy()[0, 0], target.data.cpu().numpy()[0, 0],
                               data_range=2)

            loss = criterion(output, target)
            prev_loss += criterion(x, target).data.cpu().numpy()
            med_loss += loss.data.cpu().numpy()
            input_and_target.set_description(
                'Loss = ' + str("{0:.3f}".format(med_loss / (i + 1)))
                + ' Loss = ' + str("{0:.3f}".format(prev_loss / (i + 1)))
                + ' Loss = ' + str("{0:.3f}".format(mean_ssim1 / (i + 1)))
                + ' Loss = ' + str("{0:.3f}".format(mean_ssim2 / (i + 1))))
        if epoch % args.save_freq:
            torch.save(model.state_dict(), os.path.join(args.output, 'model_epoch{}.h5'.format(epoch)))

    torch.save(model.state_dict(), os.path.join(args.output, 'model_latest.h5'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transform CM whole-slides to H&E.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data-root', type=str, required=True, help='directory with scan crops')
    parser.add_argument('-o', '--output', required=True, help='output directory')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--model', default='log_add', help='model name.')
    parser.add_argument('--criterion', default='mse', help='loss criterion.')
    parser.add_argument('--optim', default='adam', help='optimizer name.')
    parser.add_argument('-l', '--learning-rate', dest='lr', type=int, default=1e-3)
    parser.add_argument('-e', '--epochs', type=int, default=100)
    parser.add_argument('--save-freq', type=int, metavar='EPOCH', default=5,
                        help='save model checkpoint after every EPOCH')
    parser.add_argument('--crop', type=int, default=256, help='size in pixels of square random crop.')
    parser.add_argument('--noise', default='gamma', choices=['gaussian', 'gamma'],
                        help='type of noise Gaussian(1, 0.2) or Gamma(1, 1).')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    if args.verbose:
        print(args)
    main(args)
