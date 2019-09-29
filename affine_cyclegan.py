import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time

import torch
import torchvision.transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn

from cyclegan.models import *
from cyclegan.utils import *
from datasets import CMCropsDataset, SimpleDataset
from transforms import (CMRandomCrop, CMRandomHorizontalFlip,
                        CMRandomVerticalFlip, CMToTensor, CMCompose,
                        VirtualStainer)


torch.manual_seed(0)
np.random.seed(0)


class AffineGenerator(nn.Module):
    """Affine transform generator implemented as a single layer 1x1 conv layer."""

    def __init__(self, input_nc, output_nc):
        super().__init__()
        self.model = nn.Conv2d(input_nc, output_nc, 1)

    def forward(self, x):
        x = self.model(x)
        # return torch.tanh(x)
        return x


class UnalignedCM2HEDataset(Dataset):
    def __init__(self, cm_root, he_root, transform_cm=None, transform_he=None):
        self.cm_dataset = CMCropsDataset(cm_root, transform=transform_cm)
        self.he_dataset = SimpleDataset(he_root, transform=transform_he)

        self.cm_to_tensor = CMToTensor()
        # self.cm_normalize = torchvision.transforms.Normalize((0.5, 0.5), (0.5, 0.5))
        self.he_to_tensor = torchvision.transforms.ToTensor()
        # self.he_normalize = torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def __getitem__(self, item):
        cm = self.cm_dataset[item % len(self.cm_dataset)]
        cm = self.cm_to_tensor(cm['R'], cm['F'])
        # cm = self.cm_normalize(cm)
        he = self.he_dataset[random.randrange(len(self.he_dataset))]
        he = self.he_to_tensor(he)
        # he = self.he_normalize(he)

        return {'CM': cm, 'HE': he}

    def __len__(self):
        return max(len(self.cm_dataset), len(self.he_dataset))


def main(opt):
    # Create sample and checkpoint directories
    os.makedirs('images/%s' % opt.dataset_name, exist_ok=True)
    os.makedirs('saved_models/%s' % opt.dataset_name, exist_ok=True)

    # Losses
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()

    cuda = torch.cuda.is_available() and not opt.no_cuda
    device = 'cuda:0' if cuda else 'cpu'

    # Calculate output of image discriminator (PatchGAN)
    patch = (1,
             opt.img_height // 2 ** opt.discriminator_blocks,
             opt.img_width // 2 ** opt.discriminator_blocks)

    # Instantiate generator and discriminator
    CM_to_HE = AffineGenerator(2, 3)
    HE_to_CM = AffineGenerator(3, 2)
    D_CM = Discriminator(in_channels=2, discriminator_blocks=opt.discriminator_blocks)
    D_HE = Discriminator(in_channels=3, discriminator_blocks=opt.discriminator_blocks)
    # Initialize weights and bias with Gareau staining technique.
    CM_to_HE.model.weight.data = torch.tensor(
        [[VirtualStainer.E[0] - 1,
          VirtualStainer.H[0] - 1],
         [VirtualStainer.E[1] - 1,
          VirtualStainer.H[1] - 1],
         [VirtualStainer.E[2] - 1,
          VirtualStainer.H[2] - 1]],
    ).reshape(CM_to_HE.model.weight.shape)
    CM_to_HE.model.bias.data.fill_(1)

    CM_to_HE.to(device)
    HE_to_CM.to(device)
    D_CM.to(device)
    D_HE.to(device)

    if opt.epoch != 0:
        # Load pretrained models
        CM_to_HE.load_state_dict(torch.load('saved_models/%s/CM-to-HE_%d.pth' % (opt.dataset_name, opt.epoch)))
        HE_to_CM.load_state_dict(torch.load('saved_models/%s/HE-to-CM_%d.pth' % (opt.dataset_name, opt.epoch)))
        D_CM.load_state_dict(torch.load('saved_models/%s/D_CM_%d.pth' % (opt.dataset_name, opt.epoch)))
        D_HE.load_state_dict(torch.load('saved_models/%s/D_HE_%d.pth' % (opt.dataset_name, opt.epoch)))
    else:
        # Initialize weights
        HE_to_CM.apply(weights_init_normal)
        D_CM.apply(weights_init_normal)
        D_HE.apply(weights_init_normal)

    # Loss weights
    lambda_cyc = opt.lambda_cycle

    # Optimizers
    optimizer_CM_to_HE = torch.optim.Adam(CM_to_HE.parameters(),
                                          lr=opt.lr / 10, betas=(opt.b1, opt.b2))
    optimizer_HE_to_CM = torch.optim.Adam(HE_to_CM.parameters(),
                                          lr=opt.lr / 10, betas=(opt.b1, opt.b2))
    optimizer_D_HE = torch.optim.Adam(D_CM.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D_CM = torch.optim.Adam(D_HE.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    # Learning rate update schedulers
    lr_scheduler_CM_to_HE = torch.optim.lr_scheduler.LambdaLR(
        optimizer_CM_to_HE,
        lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
    )
    lr_scheduler_HE_to_CM = torch.optim.lr_scheduler.LambdaLR(
        optimizer_HE_to_CM,
        lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
    )
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_HE,
        lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
    )
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_CM,
        lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
    )

    # Buffers of previously generated samples
    fake_CM_buffer = ReplayBuffer()
    fake_HE_buffer = ReplayBuffer()

    # Image transformations
    relation = 0.65
    transforms_HE = torchvision.transforms.Compose([
        torchvision.transforms.Resize(int(1024 / relation)),
        torchvision.transforms.RandomCrop((int(opt.img_height), int(opt.img_width))),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomVerticalFlip()
    ])

    transforms_CM = CMCompose([
        CMRandomCrop(opt.img_height, opt.img_width),
        CMRandomHorizontalFlip(),
        CMRandomVerticalFlip()
    ])

    # val_transforms_HE = [torchvision.transforms.RandomCrop((int(opt.img_height), int(opt.img_width))),
    #                      transforms.Resize((1024, 1024)),]

    # val_transforms_CM = [torchvision.transforms.RandomCrop((int(opt.img_height), int(opt.img_width))),
    #                      torchvision.transforms.RandomHorizontalFlip(),
    #                      torchvision.transforms.RandomVerticalFlip(),
    #                      torchvision.transforms.ToTensor(),
    #                      torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    # Training data loader
    dataset = UnalignedCM2HEDataset(opt.cm_data_root, opt.he_data_root,
                                    transforms_CM, transforms_HE)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True,
                            num_workers=opt.n_cpu)
    # Test data loader
    val_dataloader = DataLoader(dataset, batch_size=4, shuffle=False,
                                num_workers=1)
    val_iterator = iter(val_dataloader)

    def sample_images(batches_done, iterator):
        """Saves a generated sample from the test set."""
        try:
            imgs = next(iterator)
        except StopIteration:
            iterator = iter(val_dataloader)
            imgs = next(iterator)
        real_CM = imgs['CM'].to(device)
        fake_HE = CM_to_HE(real_CM)
        real_HE = imgs['HE'].to(device)
        # fake_CM = HE_to_CM(real_HE)
        img_sample = torch.cat((fake_HE.data, real_HE.data), 0)
        save_image(img_sample, 'images/%s/%s.png' % (opt.dataset_name, batches_done),
                   nrow=4, normalize=True)

    # ----------
    #  Training
    # ----------

    prev_time = time.time()
    for epoch in range(opt.epoch, opt.n_epochs):
        for i, batch in enumerate(dataloader):

            # Set model input
            real_cm = batch['CM'].to(device)
            real_he = batch['HE'].to(device)

            # Adversarial ground truths
            valid = torch.tensor(np.ones((real_cm.size(0), *patch)), requires_grad=False,
                                 device=device, dtype=torch.float)
            fake = torch.tensor(np.zeros((real_cm.size(0), *patch)), requires_grad=False,
                                device=device, dtype=torch.float)

            # ------------------
            #  Train Generators
            # ------------------

            optimizer_HE_to_CM.zero_grad()
            optimizer_CM_to_HE.zero_grad()

            # GAN loss
            fake_he = CM_to_HE(real_cm)
            loss_GAN_AB = criterion_GAN(D_HE(fake_he), valid)
            fake_cm = HE_to_CM(real_he)
            loss_GAN_BA = criterion_GAN(D_CM(fake_cm), valid)

            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

            # Cycle loss
            recov_cm = HE_to_CM(fake_he)
            loss_cycle_A = criterion_cycle(recov_cm, real_cm)
            recov_he = CM_to_HE(fake_cm)
            loss_cycle_B = criterion_cycle(recov_he, real_he)

            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

            # Total loss
            loss_G = loss_GAN + lambda_cyc * loss_cycle

            loss_G.backward()
            optimizer_HE_to_CM.step()
            optimizer_CM_to_HE.step()

            # -----------------------
            #  Train Discriminator A
            # -----------------------

            optimizer_D_HE.zero_grad()

            # Real loss
            loss_real = criterion_GAN(D_CM(real_cm), valid)
            # Fake loss (on batch of previously generated samples)
            fake_A_ = fake_CM_buffer.push_and_pop(fake_cm)
            loss_fake = criterion_GAN(D_CM(fake_A_.detach()), fake)
            # Total loss
            loss_D_A = (loss_real + loss_fake) / 2

            loss_D_A.backward()
            optimizer_D_HE.step()

            # -----------------------
            #  Train Discriminator B
            # -----------------------

            optimizer_D_CM.zero_grad()

            # Real loss
            loss_real = criterion_GAN(D_HE(real_he), valid)
            # Fake loss (on batch of previously generated samples)
            fake_B_ = fake_HE_buffer.push_and_pop(fake_he)
            loss_fake = criterion_GAN(D_HE(fake_B_.detach()), fake)
            # Total loss
            loss_D_B = (loss_real + loss_fake) / 2

            loss_D_B.backward()
            optimizer_D_CM.step()

            loss_D = (loss_D_A + loss_D_B) / 2

            # --------------
            #  Log Progress
            # --------------

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + i
            batches_left = opt.n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            sys.stdout.write('\r[Epoch %d/%d] [Batch %d/%d] '
                             '[D loss: %f] [G loss: %f, adv: %f, cycle: %f] ETA: %s' %
                             (epoch, opt.n_epochs, i, len(dataloader),
                              loss_D.item(), loss_G.item(), loss_GAN.item(),
                              loss_cycle.item(), time_left))

            # If at sample interval save image
            if batches_done % opt.sample_interval == 0:
                sample_images(batches_done, val_iterator)

        # Update learning rates
        lr_scheduler_CM_to_HE.step()
        lr_scheduler_HE_to_CM.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(CM_to_HE.state_dict(), 'saved_models/%s/CM-to-HE_%d.pth' % (opt.dataset_name, epoch))
            torch.save(HE_to_CM.state_dict(), 'saved_models/%s/HE-to-CM_%d.pth' % (opt.dataset_name, epoch))
            torch.save(D_CM.state_dict(), 'saved_models/%s/D_CM_%d.pth' % (opt.dataset_name, epoch))
            torch.save(D_HE.state_dict(), 'saved_models/%s/D_HE_%d.pth' % (opt.dataset_name, epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
    parser.add_argument('--n-epochs', type=int, default=10, help='number of epochs of training')
    parser.add_argument('--cm-data-root', required=True, help='CM dataset path')
    parser.add_argument('--he-data-root', required=True, help='HE dataset path')
    parser.add_argument('--discriminator-blocks', type=int, default=2, help='number of discriminator blocks')
    parser.add_argument('--batch-size', type=int, default=8, help='size of the batches')
    parser.add_argument('--lr', type=float, default=0.001, help='adam: learning rate')
    parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of second order momentum of gradient')
    parser.add_argument('--decay-epoch', type=int, default=8, help='epoch from which to start lr decay')
    parser.add_argument('--lambda-cycle', type=float, default=10., help='cycle loss weight')
    parser.add_argument('--img-height', type=int, default=256, help='size of image height')
    parser.add_argument('--img-width', type=int, default=256, help='size of image width')
    parser.add_argument('--sample-interval', type=int, default=100,
                        help='interval between sampling images from generators')
    parser.add_argument('--checkpoint-interval', type=int, default=1, help='interval between saving model checkpoints')
    parser.add_argument('--n-cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    opt = parser.parse_args()
    opt.dataset_name = 'affine_conf'

    if opt.verbose:
        print(opt)

    main(opt)
