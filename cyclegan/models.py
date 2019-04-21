import torch.nn as nn
import torch.nn.functional as F
import torch


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


##############################
#           RESNET
##############################

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features),
                      nn.ReLU(inplace=True),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class GeneratorResNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, res_blocks=9):
        super(GeneratorResNet, self).__init__()

        # Initial convolution block
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(in_channels, 64, 7),
                 nn.InstanceNorm2d(64),
                 nn.ReLU(inplace=True)]

        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        for _ in range(res_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(64, out_channels, 7),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_down=2):
        super(GeneratorUNet, self).__init__()
        self.num_down = num_down

        def downsampling_block(in_features, out_features, normalize=True):
            """Returns downsampling module of each generator block."""
            layers = [nn.Conv2d(in_features, out_features, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_features))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        def upsampling_block(in_features, out_features, normalize=True):
            """Returns upsampling layers of each generator block."""
            layers = [nn.ConvTranspose2d(in_features, out_features, 3,
                                         stride=2, padding=1, output_padding=1)
                      ]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_features))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        # Initial convolution block
        self.first = nn.Sequential(nn.ReflectionPad2d(3),
                                   nn.Conv2d(in_channels, 64, 7),
                                   nn.InstanceNorm2d(64),
                                   nn.ReLU(inplace=True))

        # Downsampling
        self.down_layers = []
        in_features = 64
        out_features = in_features * 2
        for i in range(self.num_down):
            self.down_layers[i] = downsampling_block(in_features, out_features)
            in_features = out_features
            out_features = in_features * 2

        # Middle
        self.middle = downsampling_block(in_features, in_features)

        # Upsampling
        self.up_layers = []
        out_features = in_features // 2
        for i in range(self.num_down):
            # multiply in_features by two counting concatenated channels
            self.up_layers[i] = upsampling_block(in_features * 2, out_features)
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        self.last = nn.Sequential(nn.ReflectionPad2d(3),
                                  nn.Conv2d(64, out_channels, 7),
                                  nn.Tanh())

    def forward(self, x):
        out_first = self.first(x)
        out_downs = [self.down_layers[0](out_first)]
        for i in range(1, self.num_down):
            out_downs.append(self.down_layers[i](out_downs[-1]))
        out_middle = self.middle(out_downs[-1])
        out_ups = [self.up_layers[0](out_middle)]
        for i in range(1, self.num_down):
            in_up = torch.cat((out_ups[-1], out_downs[-i]), dim=1)
            out_ups.append(self.up_layers[i](in_up))
        return self.last(out_ups[-1])


class GeneratorUResNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, res_blocks=9):
        super(GeneratorUResNet, self).__init__()

        # Initial convolution block
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(in_channels, 64, 7),
                 nn.InstanceNorm2d(64),
                 nn.ReLU(inplace=True)]

        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        for _ in range(res_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(64, out_channels, 7),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


##############################
#        Discriminator
##############################

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block."""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)
