# Copyright (c) 2022 Carl Zeiss AG – All Rights Reserved.
# ZEISS, ZEISS.com are registered trademarks of Carl Zeiss AG

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['UNet']


class UNet(nn.Module):

    def __init__(self, num_channels, num_classes, depth=4, initial_filter_count=64, bilinear=True):
        super(UNet, self).__init__()

        self.num_channels = num_channels
        self.num_classes = num_classes
        self.depth = depth
        self.initial_filter_count = initial_filter_count
        self.bilinear = bilinear

        factor = 2 if bilinear else 1

        filter_count = initial_filter_count

        encoder_blocks = []
        encoder_blocks.append(DoubleConv(num_channels, filter_count))
        for d in range(depth):
            if d < depth - 1:
                encoder_blocks.append(Down(filter_count, 2 * filter_count))
            else:
                encoder_blocks.append(Down(filter_count, (2 * filter_count) // factor))
            filter_count *= 2
        self.encoder_blocks = nn.Sequential(*encoder_blocks)

        decoder_blocks = []
        for d in range(depth):
            if d < depth - 1:
                decoder_blocks.append(Up(filter_count, filter_count // 2 // factor, bilinear))
            else:
                decoder_blocks.append(Up(filter_count, filter_count // 2, bilinear))
            filter_count //= 2
        self.decoder_blocks = nn.Sequential(*decoder_blocks)

        self.outc = OutputConvolution(filter_count, num_classes)

    def forward(self, x):
        xs = []
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
            xs.append(x)

        xs.reverse()
        xs = xs[1:]

        for decoder_block, x_skip in zip(self.decoder_blocks, xs):
            x = decoder_block(x, x_skip)

        logits = self.outc(x)

        return logits


class DoubleConv(nn.Module):
    """Module combining Conv -> BN -> ReLU -> Conv -> BN -> ReLU."""

    def __init__(self, num_input_channels, num_output_channels, num_middle_channels=None):
        super().__init__()

        if not num_middle_channels:
            num_middle_channels = num_output_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(
                num_input_channels,
                num_middle_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                num_middle_channels,
                num_output_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Module combining downscaling and DoubleConvolution."""

    def __init__(self, num_input_channels, num_output_channels):
        super().__init__()

        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2),
                                          DoubleConv(num_input_channels, num_output_channels))

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Module combining upscaling and DoubleConvolution."""

    def __init__(self, num_input_channels, num_output_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(num_input_channels, num_output_channels, num_input_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(num_input_channels,
                                         num_input_channels // 2,
                                         kernel_size=2,
                                         stride=2)
            self.conv = DoubleConv(num_input_channels, num_output_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)


class OutputConvolution(nn.Module):

    def __init__(self, num_input_channels, num_output_channels):
        super(OutputConvolution, self).__init__()

        self.conv = nn.Conv2d(num_input_channels, num_output_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
