from __future__ import print_function, division

import torch.nn as nn
import torch.nn.functional as F
import torch

import sys

class ConvModel(nn.Module):
    def __init__(self, noise_magnitude=0.2):
        super(ConvModel, self).__init__()

        self._noise_magnitude = noise_magnitude

        self._conv_part = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.PReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.PReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.PReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.PReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.PReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.PReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 1, 3, padding=1),
        )

        self._tanh = nn.Tanh()

    def forward(self, x):
        clean = x * (1 + self._tanh(self._conv_part(x)))
        clean = self._tanh(clean)

        return clean
