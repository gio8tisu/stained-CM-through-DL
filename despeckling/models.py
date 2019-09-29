from __future__ import print_function, division

import torch.nn as nn
import torch

SAFE_LOG_EPSILON = 1E-5  # small number to avoid log(0).
SAFE_DIV_EPSILON = 1E-5  # small number to avoid division by zero.


class ResModel(nn.Module):
    """Model with residual/skip connection."""

    def __init__(self, sub_module, skip_connection=lambda x, y: x + y):
        """

        :param sub_module: model between input and skip connection.
        :param skip_connection: operation to do in skip connection.
        """
        super(ResModel, self).__init__()

        self.skip_connection = skip_connection

        self.noise_removal_block = sub_module

    def forward(self, x):
        clean = self.skip_connection(x, self.noise_removal_block(x))
        # clean = self._tanh(clean)

        return clean


class BasicConv(nn.Module):
    """Series of convolution layers keeping the same image shape."""

    def __init__(self, in_channels=1, n_layers=6, n_filters=64, kernel_size=3):
        super(BasicConv, self).__init__()
        model = [nn.Sequential(nn.Conv2d(in_channels, n_filters, kernel_size,
                                         padding=kernel_size // 2),
                               nn.BatchNorm2d(n_filters),
                               nn.PReLU())
                 ]
        for _ in range(n_layers - 1):
            model += [nn.Sequential(nn.Conv2d(n_filters, n_filters, kernel_size,
                                              padding=kernel_size // 2),
                                    nn.BatchNorm2d(n_filters),
                                    nn.PReLU())
                      ]
        model += [nn.Conv2d(n_filters, in_channels, 1)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class DilatedConv(nn.Module):
    """Series of convolution layers with dilation 2 keeping the same image shape."""

    def __init__(self, in_channels=1, n_layers=6, n_filters=64, kernel_size=3):
        super(DilatedConv, self).__init__()
        model = [nn.Sequential(nn.Conv2d(in_channels, n_filters, kernel_size,
                                         padding=kernel_size // 2 * 2, dilation=2),
                               nn.BatchNorm2d(n_filters),
                               nn.PReLU())
                 ]
        for _ in range(n_layers - 1):
            model += [nn.Sequential(nn.Conv2d(n_filters, n_filters, kernel_size,
                                              padding=kernel_size // 2 * 2, dilation=2),
                                    nn.BatchNorm2d(n_filters),
                                    nn.PReLU())
                      ]
        model += [nn.Conv2d(n_filters, in_channels, 1)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class LogAddDespeckle(nn.Module):
    """Apply log to pixel values, residual block with addition, apply exponential."""

    def __init__(self, n_layers=6, n_filters=64, kernel_size=3, apply_sigmoid=True):
        super(LogAddDespeckle, self).__init__()
        conv = BasicConv(in_channels=1, n_layers=n_layers, n_filters=n_filters,
                         kernel_size=kernel_size)
        self.remove_noise = ResModel(conv, skip_connection=lambda x, y: x + y)
        self.apply_sigmoid = apply_sigmoid

    def forward(self, x):
        log_x = (x + SAFE_LOG_EPSILON).log()
        clean_log_x = self.remove_noise(log_x)
        clean_x = clean_log_x.exp()
        if self.apply_sigmoid:
            return torch.sigmoid(clean_x)
        return clean_x


class DilatedLogAddDespeckle(nn.Module):
    """Apply log to pixel values, residual block with addition, apply exponential."""

    def __init__(self, n_layers=6, n_filters=64, kernel_size=3, apply_sigmoid=True):
        super(DilatedLogAddDespeckle, self).__init__()
        conv = DilatedConv(in_channels=1, n_layers=n_layers, n_filters=n_filters,
                           kernel_size=kernel_size)
        self.remove_noise = ResModel(conv, skip_connection=lambda x, y: x + y)
        self.apply_sigmoid = apply_sigmoid

    def forward(self, x):
        log_x = (x + SAFE_LOG_EPSILON).log()
        clean_log_x = self.remove_noise(log_x)
        clean_x = clean_log_x.exp()
        if self.apply_sigmoid:
            return torch.sigmoid(clean_x)
        return clean_x


class LogSubtractDespeckle(nn.Module):
    """Apply log to pixel values, residual block with subtraction, apply exponential."""

    def __init__(self, n_layers=6, n_filters=64, kernel_size=3, apply_sigmoid=True):
        super(LogSubtractDespeckle, self).__init__()
        conv = BasicConv(in_channels=1, n_layers=n_layers, n_filters=n_filters,
                         kernel_size=kernel_size)
        self.remove_noise = ResModel(conv, skip_connection=lambda x, y: x - y)
        self.apply_sigmoid = apply_sigmoid

    def forward(self, x):
        log_x = (x + SAFE_LOG_EPSILON).log()
        clean_log_x = self.remove_noise(log_x)
        clean_x = clean_log_x.exp()
        if self.apply_sigmoid:
            return torch.sigmoid(clean_x)
        return clean_x


class MultiplyDespeckle(nn.Module):
    """Residual block with multiplication."""

    def __init__(self, n_layers=6, n_filters=64, kernel_size=3, apply_sigmoid=True):
        super(MultiplyDespeckle, self).__init__()
        conv = BasicConv(in_channels=1, n_layers=n_layers, n_filters=n_filters,
                         kernel_size=kernel_size)
        self.remove_noise = ResModel(conv, skip_connection=lambda x, y: x * y)
        self.apply_sigmoid = apply_sigmoid

    def forward(self, x):
        clean_x = self.remove_noise(x)
        if self.apply_sigmoid:
            return torch.sigmoid(clean_x)
        return clean_x


class DivideDespeckle(nn.Module):
    """Residual block with division."""

    def __init__(self, n_layers=6, n_filters=64, kernel_size=3, apply_sigmoid=True):
        super(DivideDespeckle, self).__init__()
        conv = BasicConv(in_channels=1, n_layers=n_layers, n_filters=n_filters,
                         kernel_size=kernel_size)
        self.remove_noise = ResModel(
            conv,
            skip_connection=lambda x, y: x / (y + SAFE_DIV_EPSILON)
        )
        self.apply_sigmoid = apply_sigmoid

    def forward(self, x):
        clean_x = self.remove_noise(x)
        if self.apply_sigmoid:
            return torch.sigmoid(clean_x)
        return clean_x
