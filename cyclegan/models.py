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


# encoder block
class DownsamplingBlock(nn.Module):
    """Returns downsampling module of each generator block.

    conv + instance norm + relu
    """
    def __init__(self, in_features, out_features, normalize=True):
        super(DownsamplingBlock, self).__init__()
        layers = [nn.Conv2d(in_features, out_features, 3,
                            stride=2, padding=1)
                  ]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_features))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# decoder block
class UpsamplingBlock(nn.Module):
    """Returns UNet upsampling layers of each generator block.

    transposed conv + instance norm + relu
    """
    def __init__(self, in_features, out_features, normalize=True):
        super(UpsamplingBlock, self).__init__()
        # multiply in_features by two because of concatenated channels.
        layers = [nn.ConvTranspose2d(in_features * 2, out_features, 3,
                                     stride=2, padding=1, output_padding=1)
                  ]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_features))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.model = nn.Sequential(*layers)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        return self.model(x)


# ResNet block
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
        self.down_activations = {}

        def get_activation(name):
            def hook(model, input, output):
                self.down_activations[name] = output
            return hook

        # Initial convolution block
        self.first = nn.Sequential(nn.ReflectionPad2d(3),
                                   nn.Conv2d(in_channels, 64, 7),
                                   nn.InstanceNorm2d(64),
                                   nn.ReLU(inplace=True))

        # Downsampling
        down_layers = []
        in_features = 64
        out_features = in_features * 2
        for i in range(self.num_down):
            down_layers.append(
                DownsamplingBlock(in_features, out_features).register_forward_hook(
                    get_activation(i)))
            in_features = out_features
            out_features = in_features * 2
        self.down_layers = nn.Sequential(*down_layers)

        # Middle
        self.middle = nn.Sequential(nn.Conv2d(in_features, in_features, 3),
                                    nn.InstanceNorm2d(in_features),
                                    nn.LeakyReLU(0.2, inplace=True))

        # Upsampling
        up_layers = []
        out_features = in_features // 2
        for _ in range(self.num_down):
            up_layers.append(UpsamplingBlock(in_features, out_features))
            in_features = out_features
            out_features = in_features // 2
        self.up_layers = nn.Sequential(*up_layers)

        # Output layer
        self.last = nn.Sequential(nn.ReflectionPad2d(3),
                                  nn.Conv2d(64, out_channels, 7),
                                  nn.Tanh())

    def forward(self, x):
        out_first = self.first(x)
        out_encoder = self.down_layers(out_first)
        # out_downs = [self.down_layers[0](out_first)]
        # for i in range(1, self.num_down):
        #     out_downs.append(self.down_layers[i](out_downs[-1]))
        out_middle = self.middle(out_encoder)
        for i, decoder_layer in enumerate(self.up_layers):
            if i == 0:
                out = decoder_layer(
                    self.down_activations[self.num_down - 1 - i], out_middle)
            else:
                out = decoder_layer(
                    self.down_activations[self.num_down - 1 - i], out)

        return self.last(out)


class GeneratorUResNet(nn.Module):
    """TODO"""
    def __init__(self):
        raise NotImplementedError


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
