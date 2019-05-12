from datasets.mnist_dataset import Dataset

from torch.utils.data import DataLoader
from models.model import ConvModel
from torch.autograd import Variable
import sys

import torch

from tqdm import tqdm

from torch.nn import MSELoss, L1Loss
from torch.optim import Adam

from skimage.filters import median

from os import listdir
from os.path import isfile, join
import numpy as np

from torchvision.utils import save_image
from sklearn.model_selection import train_test_split


def files(path='/home/marccombalia/data/'):
    files = [join(path, f) for f in listdir(path) if isfile(join(path, f))]

    train_files, test_files = train_test_split(files, test_size=0.2, shuffle=True, random_state=0)

    return train_files, test_files


def main():

    train_files, test_files = files()

    train_files = train_files[:int(len(train_files))]

    train_dataset = Dataset(train_files)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)

    test_dataset = Dataset(test_files)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=4)

    model = ConvModel().cuda()
    criterion = MSELoss().cuda()

    optimizer = Adam(params=model.parameters(), lr=1e-3)

    for epoch in range(10):
        model.train()
        med_loss = 0
        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for i, (x_batch, y) in pbar:
            x_batch, y = Variable(x_batch).cuda(), Variable(y).cuda()

            optimizer.zero_grad()
            output = model(x_batch)

            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            med_loss += loss.data.cpu().numpy()

            pbar.set_description('Loss = ' + str("{0:.3f}".format(med_loss/(i+1))))

        model.eval()
        pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
        med_loss = 0
        prev_loss = 0
        mean_ssim1 = 0
        mean_ssim2 = 0

        from skimage.measure import compare_ssim as ssim

        for i, (x_batch, y) in pbar:
            x_batch, y = Variable(x_batch, requires_grad=False).cuda(), Variable(y, requires_grad=False).cuda()
            output = model(x_batch)

            output = output.data.cpu().numpy()[0, 0]
            aux_img = x_batch.data.cpu().numpy()[0, 0]

            # output *= (np.max(aux_img) - np.min(aux_img)) / (np.max(output) - np.min(output))

            output += 1
            output /= 2
            output *= 255
            output = output.astype(np.uint8)

            output = median(output)
            output = (output / 255.0 - 0.5) * 2
            output = torch.FloatTensor([[output]]).cuda()

            mean_ssim1 += ssim(output.data.cpu().numpy()[0, 0], y.data.cpu().numpy()[0, 0],
                              data_range=2)

            mean_ssim2 += ssim(x_batch.data.cpu().numpy()[0, 0], y.data.cpu().numpy()[0, 0],
                               data_range=2)

            loss = criterion(output, y)
            prev_loss += criterion(x_batch, y).data.cpu().numpy()
            med_loss += loss.data.cpu().numpy()
            pbar.set_description('Loss = ' + str("{0:.3f}".format(med_loss / (i + 1))) + ' Loss = ' + str("{0:.3f}".format(prev_loss / (i + 1)))
                                 + ' Loss = ' + str("{0:.3f}".format(mean_ssim1 / (i + 1)))
                                 + ' Loss = ' + str("{0:.3f}".format(mean_ssim2 / (i + 1))))

        continue
        output = model(x_batch)
        output = output.data

        result = torch.cat([x_batch, output, y])
        result += 1
        result /= 2
        save_image(result, 'test'+str(epoch)+'.png', nrow=len(output))

    torch.save(model.state_dict(), 'model_dump.h5')


if __name__ == '__main__':
    main()
