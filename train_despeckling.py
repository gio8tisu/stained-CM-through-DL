import argparse
import os.path
import pickle

import torch
from torch.utils.data import DataLoader, random_split
from torch.nn import MSELoss, L1Loss
from torch.optim import Adam
import numpy as np
from skimage.measure import compare_ssim as ssim
import tqdm

from datasets import NoisyScansDataset
from despeckling import models

# torch.backends.cudnn.enabled = False


def main(args):
    cuda = True if torch.cuda.is_available() else False
    device = torch.device("cuda") if cuda else torch.device("cpu")

    # Define dataset.
    if args.noise == 'gaussian':
        noise_args = {'random_variable': np.random.normal,
                      'loc': 1, 'scale': 0.1}
    elif args.noise == 'gamma':
        noise_args = {'random_variable': np.random.gamma,
                      'shape': 1, 'scale': 1}
    # dataset returns (noisy, clean) tuple
    dataset = NoisyScansDataset(args.data_root, 'F', noise_args, apply_random_crop=(not args.no_crop))
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4)

    # Define model and loss criterion.
    model = get_model(args.model, args.layers)
    if args.criterion == 'mse':
        criterion = MSELoss()
    elif args.criterion == 'l1':
        criterion = L1Loss()
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    # Define optimizer.
    if args.optim == 'adam':
        optimizer = Adam(params=model.parameters(), lr=args.lr)
    else:
        raise NotImplementedError(args.optim + 'optimizer is not supported.')

    # Training process.
    loss_hist = list()  # list of (epoch, train loss)
    loss_hist_eval = list()  # list of (epoch, validation loss)
    ssim_hist_eval = list()  # list of (epoch, validation SSIM)
    for epoch in range(args.epochs):
        # TRAINING.
        model.train()

        input_and_target = enumerate(train_dataloader)
        if args.verbose:
            print('Epoch {} of {}'.format(epoch, args.epochs - 1))
            input_and_target = tqdm.tqdm(input_and_target, total=len(train_dataloader))

        med_loss = 0
        for i, (x_batch, target_batch) in input_and_target:
            x_batch, target_batch = x_batch.float().to(device), target_batch.float().to(device)

            optimizer.zero_grad()
            output_batch = model(x_batch)

            loss = criterion(output_batch, target_batch)
            loss.backward()
            optimizer.step()

            med_loss += loss.data.cpu().numpy()

            if args.verbose:
                input_and_target.set_description('Train loss = {0:.3f}'.format(loss))
        loss_hist.append((epoch, med_loss / (i + 1)))

        # VALIDATION.
        if args.verbose:
            print('Validation:')
        model.eval()
        with model.no_grad():
            input_and_target = enumerate(val_dataloader)
            if args.verbose:
                input_and_target = tqdm.tqdm(input_and_target, total=len(val_dataloader))

            med_loss_eval = 0
            prev_loss_eval = 0
            for i, (x_batch, target_batch) in input_and_target:
                x_batch, target_batch = x_batch.float().to(device), target_batch.float().to(device)
                output_batch = model(x_batch)
                loss = criterion(output_batch, target_batch)
                med_loss_eval += loss.data.cpu().numpy()
                prev_loss_eval = criterion(x_batch, target_batch).data.cpu().numpy()

                ssim_input = compute_ssim(x_batch, target_batch)
                ssim_output = compute_ssim(output_batch, target_batch, args.median)

                if args.verbose:
                    input_and_target.set_description(
                        'Output loss = {0:.3f}'.format(loss)
                        + ' Input loss = {0:.3f}'.format(prev_loss_eval)
                        + ' Input SSIM = {0:.3f}'.format(ssim_input / args.batch_size)
                        + ' Output SSIM = {0:.3f}'.format(ssim_output / args.batch_size))
        loss_hist_eval.append((epoch, med_loss_eval / (i + 1)))
        if epoch % args.save_period:
            torch.save(model.state_dict(), os.path.join(args.output, 'model_epoch{}.h5'.format(epoch)))

    torch.save(model.state_dict(), os.path.join(args.output, 'model_latest.h5'))
    with open(os.path.join(args.output, 'train_loss_hist.pkl'), 'wb') as f:
        pickle.dump(loss_hist, f, pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(args.output, 'valid_loss_hist.pkl'), 'wb') as f:
        pickle.dump(loss_hist_eval, f, pickle.HIGHEST_PROTOCOL)


def compute_ssim(noisy_batch, clean_batch, median_filter=False):
    # iterate over batch to compute SSIM
    ssim_sum = 0
    for noisy, clean in zip(noisy_batch[:, 0], clean_batch[:, 0]):
        noisy = noisy.data.cpu().numpy()

        if median_filter:
            noisy = (noisy + 1) / 2 * 255
            noisy = noisy.astype(np.uint8)
            noisy = np.median(noisy)
            noisy = (noisy / 255.0 - 0.5) * 2

        ssim_sum += ssim(noisy, clean.data.cpu().numpy(), data_range=2)
    return ssim_sum


def get_model(model_str, num_layers):
    """return nn.Module based on model_str.

    TODO: get model class with importlib library.
    """
    if model_str == 'log_add':
        return models.LogAddDespeckle(num_layers)
    elif model_str == 'log_subtract':
        return models.LogSubtractDespeckle(num_layers)
    elif model_str == 'multiply':
        return models.MultiplyDespeckle(num_layers)
    elif model_str == 'divide':
        return models.DivideDespeckle(num_layers)
    else:
        raise NotImplementedError(model_str + 'model does not exist.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train despeckling network.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data-root', type=str, required=True, help='directory with scan crops')
    parser.add_argument('-o', '--output', required=True, help='output directory')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--model', default='log_add', help='model name.')
    parser.add_argument('--layers', default=6, type=int, help='number of convolutional layers.')
    parser.add_argument('--criterion', default='mse', choices=['gaussian', 'gamma'], help='loss criterion.')
    parser.add_argument('--optim', default='adam', help='optimizer name.')
    parser.add_argument('-l', '--learning-rate', dest='lr', type=int, default=1e-4)
    parser.add_argument('-e', '--epochs', type=int, default=200)
    parser.add_argument('--save-period', type=int, metavar='EPOCH', default=5,
                        help='save model checkpoint after every EPOCH')
    parser.add_argument('--no-crop', action='store_true', help='do not apply 256x256 random crop')
    parser.add_argument('--noise', default='gamma', choices=['gaussian', 'gamma'],
                        help='type of noise Gaussian(1, 0.2) or Gamma(1, 1).')
    parser.add_argument('--median', action='store_true', help='apply median filter on validation')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    if args.verbose:
        print(args)
    if not os.path.isdir(args.output):
        if args.verbose:
            print('Creating output directory.')
        os.mkdir(args.output)
    main(args)
