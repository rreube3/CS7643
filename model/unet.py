import torch
import torch.nn as nn
from typing import List

# Per https://arxiv.org/pdf/1505.04597.pdf stride = 2, kernel_size = 2
DECONV_KERNEL_SIZE = 2
DECONV_STRIDE = 2


class ConvBlock(nn.Module):
    """U-Net Convolutional Block"""

    def __init__(self,
                 num_channels_in: int,
                 num_channels_out: int,
                 kernel_size: int = 3,
                 padding: int = 1):
        """
        Initialize U-Net Convolutional block

        :param num_channels_in: Number of channels into the convolutional block
        :param num_channels_out: Number of channels out of the convolutional block
        :param kernel_size: Kernel size of the convolutions
        :param padding: Padding of the convolutions
        """
        # TODO: kernel_size = 3 and padding = 1 will result in the same size image - won't need to crop in the deconv?
        # Or is this only for image sizes of powers of 2?
        super().__init__()
        # A convolutional block is Conv2d -> BatchNorm2d -> ReLU -> Conv2d -> BatchNorm2d -> ReLU
        self.conv_1 = nn.Conv2d(num_channels_in, num_channels_out, kernel_size=kernel_size, padding=padding)
        self.batch_norm_1 = nn.BatchNorm2d(num_channels_out)
        self.relu_1 = nn.ReLU()
        self.conv_2 = nn.Conv2d(num_channels_out, num_channels_out, kernel_size=kernel_size, padding=padding)
        self.batch_norm_2 = nn.BatchNorm2d(num_channels_out)
        self.relu_2 = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward function of the U-Net convolutional block

        :param x: input tensor [BATCH, NUM_CHANNELS_IN, HEIGHT, WIDTH]
        :return: output tensor [BATCH, NUM_CHANNELS_OUT, HEIGHT, WIDTH]
        """
        x = self.relu_1(self.batch_norm_1(self.conv_1(x)))
        x = self.relu_2(self.batch_norm_2(self.conv_2(x)))
        return x


class UnetEncoder(nn.Module):
    """U-Net Encoder"""

    def __init__(self,
                 num_channels_in: int = 3,
                 hidden_channels: List = (64, 128, 256, 512, 1024),
                 kernel_size: int = 3,
                 padding: int = 1,
                 dropout: float = 0.2):
        """
        Initialize U-Net Encoder

        :param num_channels_in: Number of channels in (images are 3)
        :param hidden_channels: A list of block channel sizes
        :param kernel_size: Kernel size of the convolutions
        :param padding: Padding used for the convolution
        """
        super().__init__()
        # Create the convolutional blocks
        self.conv_blocks = nn.ModuleList()
        cur_num_in_channels = num_channels_in
        for cur_num_out_channels in hidden_channels:
            self.conv_blocks.append(ConvBlock(cur_num_in_channels,
                                              cur_num_out_channels,
                                              kernel_size=kernel_size,
                                              padding=padding))
            cur_num_in_channels = cur_num_out_channels
        # Create the max pooling layer
        self.max_pool = nn.MaxPool2d(kernel_size=DECONV_KERNEL_SIZE, stride=DECONV_STRIDE)
        self.dropper = nn.Dropout2d(p=dropout)

    def forward(self, x: torch.Tensor) -> (torch.Tensor, List):
        """
        Forward function of the U-Net Encoder

        :param x: input tensor [BATCH, NUM_CHANNELS_IN, HEIGHT, WIDTH]
        :return: A tuple containing the output tensor
        [BATCH, NUM_CHANNELS_OUT[-1], HEIGHT / 2**(len(hidden_channels) - 1), WIDTH / 2**(len(hidden_channels) - 1)] and
        a list of skip connections (output from each convolutional block of the encoder)
        """
        skip_connections = []
        for i, conv_block in enumerate(self.conv_blocks):
            x = conv_block(x)
            skip_connections.append(x)
            if i < len(self.conv_blocks) - 1:
                x = self.max_pool(x)

        return self.dropper(x), skip_connections


class DeconvBlock(nn.Module):
    """U-Net Deconv Block"""

    def __init__(self,
                 num_channels_in: int,
                 num_channels_out: int,
                 ):
        """
        Initialize the U-Net Deconv Block

        :param num_channels_in: Number of channels in
        :param num_channels_out: Number of channels out
        """
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(num_channels_in,
                                                 num_channels_out,
                                                 kernel_size=DECONV_KERNEL_SIZE,
                                                 stride=DECONV_STRIDE)
        self.conv_block = ConvBlock(int(DECONV_KERNEL_SIZE * num_channels_out), num_channels_out)

    def forward(self, x: torch.Tensor, skip_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward function of the U-Net deconv block

        :param x: Input tensor [BATCH, NUM_CHANNELS_IN, HEIGHT, WIDTH]
        :param skip_tensor: Skip tensor [BATCH, NUM_CHANNELS_IN, HEIGHT, WIDTH] (i.e. output from an encoder conv block)
        :return: Output tensor of [BATCH, NUM_CHANNELS_IN, 2 * HEIGHT, 2 * WIDTH]
        """
        x = self.conv_transpose(x)
        x = torch.cat([x, skip_tensor], axis=1)
        x = self.conv_block(x)
        return x


class UnetDecoder(nn.Module):
    """U-Net Decoder"""

    def __init__(self,
                 hidden_channels: List = (64, 128, 256, 512, 1024)):
        """
        Initialize the U-Net Decoder

        :param hidden_channels: Number of channels for each convolutional block of the U-Net
        """
        super().__init__()
        self.deconv_blocks = nn.ModuleList()
        cur_num_out_channels = hidden_channels[0]
        for cur_num_in_channels in hidden_channels[1:]:
            self.deconv_blocks.append(DeconvBlock(cur_num_in_channels,
                                                  cur_num_out_channels))
            cur_num_out_channels = cur_num_in_channels

    def forward(self, x: torch.Tensor, skip_connections: List) -> torch.Tensor:
        """
        Forward function of the U-Net Decoder

        :param x: Input tensor [BATCH, NUM_CHANNELS_IN, HEIGHT, WIDTH]
        :param skip_connections: Skip connections of each of the convolutional blocks
        :return: Output tensor
        [BATCH, NUM_CHANNELS_IN, HEIGHT * 2**(len(hidden_channels) - 1), WIDTH * 2**(len(hidden_channels) - 1)]
        """
        for i in range(len(self.deconv_blocks) - 1, -1, -1):
            x = self.deconv_blocks[i](x, skip_connections[i])

        return x


class Unet(nn.Module):
    """U-Net!"""

    def __init__(self,
                 num_channels_in: int = 4,
                 num_classes: int = 1,
                 hidden_channels: List = (64, 128, 256, 512, 1024),
                 kernel_size: int = 3,
                 padding: int = 1,
                 dropout: float = 0.2):
        """
        Initialize U-Net Convolutional block

        :param num_channels_in: Number of channels into the convolutional block
        :param num_classes: Number of classes to classify
        :param hidden_channels: A list of block channel sizes
        :param kernel_size: Kernel size of the convolutions
        :param padding: Padding of the convolutions
        """
        super().__init__()
        # Create the encoder
        self.encoder = UnetEncoder(num_channels_in=num_channels_in,
                                   hidden_channels=hidden_channels,
                                   kernel_size=kernel_size,
                                   padding=padding,
                                   dropout=dropout)
        # Create the decoder
        self.decoder = UnetDecoder(hidden_channels=hidden_channels)

        # Output layer
        self.classifier = nn.Conv2d(in_channels=hidden_channels[0],
                                    out_channels=num_classes,
                                    kernel_size=kernel_size,
                                    padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward function of the U-Net that calls the encoder, decoder, and classification head

        :param x: Input Tensor [BATCH, NUM_CHANNELS_IN, HEIGHT, WIDTH]
        :return: Output Tensor [BATCH, NUM_CLASSES, HEIGHT, WIDTH]
        """
        # Encoder's forward
        x, skip_connections = self.encoder(x)

        # Decoder
        x = self.decoder(x, skip_connections)

        # Run the classification head
        return self.classifier(x)
