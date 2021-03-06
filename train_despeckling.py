import argparse
import os.path
import pickle

import torch
from torch.utils.data import DataLoader, random_split
from torch.nn import MSELoss, L1Loss
from torch.optim import Adam
from torchvision import transforms
import numpy as np
from skimage.measure import compare_ssim as ssim
import tqdm

from datasets import CMCropsDataset
from despeckling import models
from transforms import MultiplicativeNoise

torch.manual_seed(0)
np.random.seed(0)


def main(opt):
    cuda = True if torch.cuda.is_available() else False
    device = torch.device('cuda') if cuda else torch.device('cpu')

    # Define dataset.
    noise_args = get_noise_args(opt.noise, L=opt.L)
    transform = transforms.Compose(
        [transforms.RandomCrop(opt.crop_size),
         transforms.ToTensor(),
         MultiplicativeNoise(**noise_args)  # returns (noisy, clean) tuple
         ]
    )
    # dataset returns (noisy, clean) tuple
    dataset = CMCropsDataset(opt.data_root, only_F=True, transform_F=transform)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=opt.batch_size, num_workers=4)

    # Define model and loss criterion.
    model = get_model(opt.model, opt.layers, opt.filters, opt.filters_size,
                      apply_sigmoid=not opt.no_sigmoid)
    if opt.criterion == 'mse':
        criterion = MSELoss()
    elif opt.criterion == 'l1':
        criterion = L1Loss()
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    # Define optimizer.
    if opt.optim == 'adam':
        optimizer = Adam(params=model.parameters(), lr=opt.lr)
    else:
        raise NotImplementedError(opt.optim + ' optimizer is not supported.')

    # Training process.
    loss_hist = list()  # list of (epoch, train loss)
    loss_hist_eval = list()  # list of (epoch, validation loss)
    ssim_hist_eval = list()  # list of (epoch, validation SSIM)
    for epoch in range(opt.epochs):
        # TRAINING.
        model.train()

        input_and_target = enumerate(train_dataloader)
        if opt.verbose:
            print('Epoch {} of {}'.format(epoch, opt.epochs - 1))
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

            if opt.verbose:
                input_and_target.set_description('Train loss = {0:.3f}'.format(loss))
        loss_hist.append((epoch, med_loss / (i + 1)))

        # VALIDATION.
        if opt.verbose:
            print('Validation:')
        model.eval()
        with torch.no_grad():
            input_and_target = enumerate(val_dataloader)
            if opt.verbose:
                input_and_target = tqdm.tqdm(input_and_target, total=len(val_dataloader))

            med_loss_eval = 0
            med_ssim_eval = 0
            for i, (x_batch, target_batch) in input_and_target:
                x_batch, target_batch = x_batch.float().to(device), target_batch.float().to(device)
                output_batch = model(x_batch)
                loss = criterion(output_batch, target_batch)
                med_loss_eval += loss.data.cpu().numpy()
                prev_loss_eval = criterion(x_batch, target_batch).data.cpu().numpy()

                ssim_input = compute_ssim(x_batch, target_batch)
                ssim_output = compute_ssim(output_batch, target_batch, opt.median)
                med_ssim_eval += ssim_output

                if opt.verbose:
                    input_and_target.set_description(
                        'Validation: '
                        + 'Output loss = {0:.3f}'.format(loss)
                        + ' Input loss = {0:.3f}'.format(prev_loss_eval)
                        + ' Input SSIM = {0:.3f}'.format(ssim_input)
                        + ' Output SSIM = {0:.3f}'.format(ssim_output))
        loss_hist_eval.append((epoch, med_loss_eval / (i + 1)))
        ssim_hist_eval.append((epoch, med_ssim_eval / (i + 1)))
        write_lc_step(epoch, loss_hist[-1][1], loss_hist_eval[-1][1], ssim_hist_eval[-1][1])
        torch.save(model.state_dict(), os.path.join(opt.output, '{}_latest.h5'.format(opt.model)))
        if epoch % opt.save_period == 0:
            torch.save(model.state_dict(), os.path.join(opt.output, '{}_epoch{}.h5'.format(opt.model, epoch)))

    torch.save(model.state_dict(), os.path.join(opt.output, 'model_latest.h5'))
    with open(os.path.join(opt.output, 'train_loss_hist.pkl'), 'wb') as f:
        pickle.dump(loss_hist, f, pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(opt.output, 'valid_loss_hist.pkl'), 'wb') as f:
        pickle.dump(loss_hist_eval, f, pickle.HIGHEST_PROTOCOL)


def get_noise_args(noise, **kwargs):
    """return dictionary with np.random function and keyword arguments.

    :param noise: (str) noise name
    """
    if noise == 'gaussian':
        return {'random_variable': np.random.normal,
                'loc': 1, 'scale': 0.1}
    elif noise == 'gamma':
        return {'random_variable': np.random.gamma,
                'shape': kwargs['L'], 'scale': 1 / kwargs['L']}
    elif noise == 'uniform':
        return {'random_variable': np.random.uniform,
                'low': 1 - 0.3464, 'high': 1 + 0.3464}
    elif noise == 'rayleigh':
        return {'random_variable': np.random.rayleigh, 'scale': 1}


def compute_ssim(noisy_batch, clean_batch, median_filter=False):
    # iterate over batch to compute mean SSIM
    ssim_sum = 0
    for noisy, clean in zip(noisy_batch[:, 0], clean_batch[:, 0]):
        noisy = noisy.data.cpu().numpy()

        if median_filter:
            noisy = (noisy + 1) / 2 * 255
            noisy = noisy.astype(np.uint8)
            noisy = np.median(noisy)
            noisy = (noisy / 255.0 - 0.5) * 2

        ssim_sum += ssim(noisy, clean.data.cpu().numpy(), data_range=2)
    return ssim_sum / noisy_batch.shape[0]


def get_model(model_str, *args, **kwargs):
    """return nn.Module based on model_str."""
    # Split by '-' character, capitalize first letter # and append 'Despeckle'.
    class_name = ''.join(map(str.capitalize, model_str.split('_'))) + 'Despeckle'
    try:
        model_class = getattr(models, class_name)
    except AttributeError:
        raise AttributeError(class_name + ' model does not exist.')
    return model_class(*args, **kwargs)


def write_lc_step(*args):
    """Write learning curve step in text file.

    Assumes arguments are passed in following order:
    epoch, train loss, validation loss, validation ssim.
    """
    with open(os.path.join(opt.output, 'learning_curve.txt'), 'a') as f:
        if args[0] == 0:  # epoch is 0
            f.write('EPOCH,TRAIN LOSS,VALIDATION LOSS,VALIDATION SSIM\n')
        # convert every argument to string, join by comma and write.
        f.write(','.join(map(str, args)) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train despeckling network.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data-root', type=str, required=True, help='directory with scan crops')
    parser.add_argument('-o', '--output', required=True, help='output directory')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--model', default='log_add', help='model name.')
    parser.add_argument('--layers', default=6, type=int, help='number of convolutional layers.')
    parser.add_argument('--filters', default=64, type=int,
                        help='number of filters on each convolution layer.')
    parser.add_argument('--filters-size', default=3, type=int,
                        help='filter/kernel size on each convolution layer')
    parser.add_argument('--no-sigmoid', action='store_true', help='no sigmoid activation in final layer')
    parser.add_argument('--criterion', default='mse', choices=['mse', 'l1'], help='loss criterion.')
    parser.add_argument('--optim', default='adam', help='optimizer name.')
    parser.add_argument('-l', '--learning-rate', dest='lr', type=float, default=1e-3)
    parser.add_argument('-e', '--epochs', type=int, default=20)
    parser.add_argument('--save-period', type=int, metavar='EPOCH', default=5,
                        help='save model checkpoint after every EPOCH')
    parser.add_argument('--crop-size', type=int, default=256, help='size of image after random crop')
    parser.add_argument('--noise', default='gamma', choices=['gaussian', 'gamma', 'uniform', 'rayleigh'],
                        help='type of noise Gaussian(1, 0.2) or Gamma(L, L).')
    parser.add_argument('-L', '--looks', metavar='L', dest='L', default=1, type=int,
                        help='number of looks used in each slide, used for gamma noise model.')
    parser.add_argument('--median', action='store_true', help='apply median filter on validation')
    parser.add_argument('-v', '--verbose', action='store_true')

    opt = parser.parse_args()
    if opt.verbose:
        print(opt)
    if not os.path.isdir(opt.output):
        if opt.verbose:
            print('Creating output directory.')
        os.mkdir(opt.output)
    main(opt)
