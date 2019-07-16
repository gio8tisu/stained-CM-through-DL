from __future__ import print_function, division

import torch.nn as nn
import torch.nn.functional as F
import torch

import sys


class ResModel(nn.Module):
    """Model with residual/skip connection."""

    def __init__(self, sub_module, skip_connection=None):
        """

        :param sub_module: model between input and skip connection.
        :param skip_connection: operation to do in skip connection.
        """
        super(ResModel, self).__init__()

        if not skip_connection:
            self.skip_connection = lambda x, y: x + y
        else:
            self.skip_connection = skip_connection

        self._conv_part = sub_module
        # self._tanh = nn.Tanh()

    def forward(self, x):
        clean = self.skip_connection(x, self._conv_part(x))
        # clean = self._tanh(clean)

        return clean


class BasicConv(nn.Module):
    """Series of convolutional layers keeping the same image shape."""

    def __init__(self, in_channels=1, n_layers=6, n_filters=64):
        super(BasicConv, self).__init__()
        model = [nn.Sequential(nn.Conv2d(in_channels, n_filters, 3, padding=1),
                               nn.BatchNorm2d(n_filters),
                               nn.PReLU())
                 ]
        for _ in range(n_layers - 1):
            model += [nn.Sequential(nn.Conv2d(n_filters, n_filters, 3, padding=1),
                                    nn.BatchNorm2d(n_filters),
                                    nn.PReLU())
                      ]
        model += [nn.Conv2d(n_filters, in_channels, 1)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class DilatedConv(nn.Module):
    """Series of convolutional layers with dilation 2 keeping the same image shape."""

    def __init__(self, in_channels=1, n_layers=6, n_filters=64):
        super(DilatedConv, self).__init__()
        model = [nn.Sequential(nn.Conv2d(in_channels, n_filters, 3, padding=1, dilation=2),
                               nn.BatchNorm2d(n_filters),
                               nn.PReLU())
                 ]
        for _ in range(n_layers - 1):
            model += [nn.Sequential(nn.Conv2d(n_filters, n_filters, 3, padding=1, dilation=2),
                                    nn.BatchNorm2d(n_filters),
                                    nn.PReLU())
                      ]
        model += [nn.Conv2d(n_filters, in_channels, 1)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class LogAddDespeckle(nn.Module):
    """Apply log to pixel values, resnet block with addition, apply exponential."""

    def __init__(self, n_layers=6, n_filters=64):
        super(LogAddDespeckle, self).__init__()
        conv = BasicConv(in_channels=1, n_layers=n_layers, n_filters=n_filters)
        self.remove_noise = ResModel(conv, skip_connection=lambda x, y: x + y)

    def forward(self, x):
        log_x = (x + 0.0001).log()
        clean_log_x = self.remove_noise(log_x)
        return clean_log_x.exp()


class DilatedLogAddDespeckle(nn.Module):
    """Apply log to pixel values, resnet block with addition, apply exponential."""

    SAVE_LOG_EPSILON = 1E-3  # small number to avoid log(0).

    def __init__(self, n_layers=6, n_filters=64):
        super(DilatedLogAddDespeckle, self).__init__()
        conv = DilatedConv(in_channels=1, n_layers=n_layers, n_filters=n_filters)
        self.remove_noise = ResModel(conv, skip_connection=lambda x, y: x + y)

    def forward(self, x):
        log_x = (x + self.SAVE_LOG_EPSILON).log()
        clean_log_x = self.remove_noise(log_x)
        return clean_log_x.exp()


class LogSubtractDespeckle(nn.Module):
    """Apply log to pixel values, resnet block with subtraction, apply exponential."""

    SAVE_LOG_EPSILON = 1E-3  # small number to avoid log(0).

    def __init__(self, n_layers=6, n_filters=64):
        super(LogSubtractDespeckle, self).__init__()
        conv = BasicConv(in_channels=1, n_layers=n_layers, n_filters=n_filters)
        self.remove_noise = ResModel(conv, skip_connection=lambda x, y: x - y)

    def forward(self, x):
        log_x = (x + self.SAVE_LOG_EPSILON).log()
        clean_log_x = self.remove_noise(log_x)
        return clean_log_x.exp()


class MultiplyDespeckle(nn.Module):
    """Resnet block with multiplication."""

    def __init__(self, n_layers=6, n_filters=64):
        super(MultiplyDespeckle, self).__init__()
        conv = BasicConv(in_channels=1, n_layers=n_layers, n_filters=n_filters)
        self.remove_noise = ResModel(conv, skip_connection=lambda x, y: x * y)
        self.last_activation = nn.Sequential(nn.BatchNorm2d(1), nn.Sigmoid())

    def forward(self, x):
        clean_x = self.remove_noise(x)
        return self.last_activation(clean_x)


class DivideDespeckle(nn.Module):
    """Resnet block with division."""

    SAVE_DIV_EPSILON = 1E-3  # small number to avoid division by zero.

    def __init__(self, n_layers=6, n_filters=64):
        super(DivideDespeckle, self).__init__()
        conv = BasicConv(in_channels=1, n_layers=n_layers, n_filters=n_filters)
        self.remove_noise = ResModel(conv,
                                     skip_connection=lambda x, y: x / (y + self.SAVE_DIV_EPSILON))
        self.last_activation = nn.Sequential(nn.BatchNorm2d(1), nn.Sigmoid())

    def forward(self, x):
        clean_x = self.remove_noise(x)
        return self.last_activation(clean_x)
