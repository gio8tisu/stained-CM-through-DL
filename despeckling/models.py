from __future__ import print_function, division

import torch.nn as nn
import torch.nn.functional as F
import torch

import sys


class ResModel(nn.Module):
    """Model with residual/skip connection."""
    def __init__(self, sub_module, skip_connection=None, noise_magnitude=0.2):
        """

        :param sub_module: model between input and skip connection.
        :param skip_connection: operation to do in skip connection.
        """
        super(ResModel, self).__init__()

        if not skip_connection:
            self.skip_connection = lambda x, y: x + y
        else:
            self.skip_connection = skip_connection

        self._noise_magnitude = noise_magnitude

        self._conv_part = sub_module

        # self._tanh = nn.Tanh()

    def forward(self, x):
        clean = self.skip_connection(x, self._conv_part(x))
        # clean = self._tanh(clean)

        return clean


class BasicConv(nn.Module):
    """Series of conv layers keeping the same image shape."""
    def __init__(self, in_channels=1, n_layers=6):
        super(BasicConv, self).__init__()
        model = [nn.Sequential(nn.Conv2d(in_channels, 64, 3, padding=1),
                               nn.PReLU(),
                               nn.BatchNorm2d(64))
                 ]
        for _ in range(n_layers - 2):
            model += [nn.Sequential(nn.Conv2d(64, 64, 3, padding=1),
                                    nn.PReLU(),
                                    nn.BatchNorm2d(64))
                           ]
        model += [nn.Conv2d(64, in_channels, 3, padding=1)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class LogSubtractDespeckle(nn.Module):
    """Apply log to pixel values, resnet with addition, apply exponential."""
    def __init__(self, n_layers=6):
        super(LogSubtractDespeckle, self).__init__()
        conv = BasicConv(in_channels=1, n_layers=n_layers)
        self.remove_noise = ResModel(conv, skip_connection=lambda x, y: x - y)

    def forward(self, x):
        log_x = x.log()
        clean_log_x = self.remove_noise(log_x)
        return clean_log_x.exp()
