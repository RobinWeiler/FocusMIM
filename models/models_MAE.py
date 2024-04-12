import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride=1, padding=0):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size, padding=padding)

        self.match_dim = nn.Conv2d(in_dim, out_dim, 1, stride=stride, padding=0)

    def forward(self, input):
        identity = input

        output = self.conv1(input)
        output = self.relu(output)
        output = self.conv2(output)

        if self.in_dim == self.out_dim:
            output += identity
        else:
            identity = self.match_dim(identity)
            output += identity

        return output

class UpConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride=1, padding=0, output_padding=1):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.upconv1 = nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.relu = nn.ReLU()
        self.upconv2 = nn.ConvTranspose2d(out_dim, out_dim, kernel_size, padding=padding, output_padding=0)

        self.match_dim = nn.ConvTranspose2d(in_dim, out_dim, 1, stride=stride, padding=0, output_padding=1)

    def forward(self, input):
        identity = input

        output = self.upconv1(input)
        output = self.relu(output)
        output = self.upconv2(output)

        if self.in_dim == self.out_dim:
            output += identity
        else:
            identity = self.match_dim(identity)
            output += identity

        return output

class CNN_MAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            ConvBlock(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            ConvBlock(32, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            ConvBlock(32, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            ConvBlock(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            ConvBlock(64, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            ConvBlock(64, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            ConvBlock(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            ConvBlock(128, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            ConvBlock(128, 128, 3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            UpConvBlock(128, 128, 3, stride=1, padding=1, output_padding=0),
            nn.ReLU(),
            UpConvBlock(128, 128, 3, stride=1, padding=1, output_padding=0),
            nn.ReLU(),
            UpConvBlock(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            UpConvBlock(64, 64, 3, stride=1, padding=1, output_padding=0),
            nn.ReLU(),
            UpConvBlock(64, 64, 3, stride=1, padding=1, output_padding=0),
            nn.ReLU(),
            UpConvBlock(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            UpConvBlock(32, 32, 3, stride=1, padding=1, output_padding=0),
            nn.ReLU(),
            UpConvBlock(32, 32, 3, stride=1, padding=1, output_padding=0),
            nn.ReLU(),
            UpConvBlock(32, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, input):
        representations = self.encoder(input)
        
        reconstructions = self.decoder(representations)

        return reconstructions, representations
