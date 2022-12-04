import torch
from torch import nn
from typing import Union, Tuple


"""REF: https://arxiv.org/pdf/2101.00535.pdf"""
"""TF/KERAS Reference Implementation: https://github.com/SharifAmit/RVGAN"""



class SeparableConv2d(nn.Module):
    """REF: https://arxiv.org/pdf/1706.03059.pdf"""
    """REF: https://stackoverflow.com/questions/65154182/implement-separableconv2d-in-pytorch"""
    def __init__(self,
                 num_channels_in: int,
                 num_channels_out: int,
                 kernel_size: Union[int, Tuple],
                 padding: Union[str, Tuple] = "same"):
        super().__init__()
        self.depthwise_conv = nn.Conv2d(num_channels_in, num_channels_in, kernel_size=kernel_size,
                                        groups=num_channels_in, bias=False, padding=padding)
        self.pointwise_conv = nn.Conv2d(num_channels_in, num_channels_out,
                                        kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


class SpatialFeatureAGgregation(nn.Module):
    def __init__(self,
                 num_channels: int):
        super().__init__()

        self.seperable_conv = SeparableConv2d(num_channels,
                                              num_channels,
                                              kernel_size=(3, 3),
                                              padding="same")
        self.batch_norm_1 = nn.BatchNorm2d(num_channels)
        self.leaky_relu_1 = nn.LeakyReLU()
        
        self.conv = nn.Conv2d(num_channels,
                              num_channels,
                              kernel_size=(3, 3),
                              padding="same")
        self.batch_norm_2 = nn.BatchNorm2d(num_channels)
        self.leaky_relu_2 = nn.LeakyReLU()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        resultant = self.seperable_conv(x)
        resultant = self.batch_norm_1(resultant)
        resultant = self.leaky_relu_1(resultant)
        resultant += torch.add(x, resultant)

        resultant = self.conv(resultant)
        resultant = self.batch_norm_2(resultant)
        resultant = self.leaky_relu_2(resultant)
        resultant = torch.add(x, resultant)
        return resultant


class EncoderBlock(nn.Module):
    def __init__(self,
                 num_channels_in: int,
                 num_channels_out: int):
        super().__init__()
        self.conv = nn.Conv2d(num_channels_in,
                              num_channels_out,
                              kernel_size=(4, 4),
                              stride=(2, 2),
                              padding=(1, 1))
        self.batch_norm = nn.BatchNorm2d(num_channels_out)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.leaky_relu(self.batch_norm(self.conv(x)))
        return x


class DecoderBlock(nn.Module):
    def __init__(self,
                 num_channels_in: int,
                 num_channels_out: int):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(num_channels_in,
                                         num_channels_out,
                                         kernel_size=(4, 4),
                                         stride=(2, 2),
                                         padding=(1, 1))
        self.batch_norm = nn.BatchNorm2d(num_channels_out)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.leaky_relu(self.batch_norm(self.deconv(x)))
        return x


class GeneratorResidualBlock(nn.Module):
    def __init__(self,
                 num_channels_in: int,
                 num_channels_out: int):
        self.reflection_padding = nn.ReflectionPad2d((1, 1))
        self.seperable_conv = SeparableConv2d(num_channels_in,
                                              num_channels_out,
                                              kernel_size=(3, 3),
                                              padding="valid")
        self.batch_norm = nn.BatchNorm2d(num_channels_out)
        self.leaky_relu = nn.LeakyReLU()

        # Branch 1
        self.branch_1_reflection_padding = nn.ReflectionPad2d((2, 2))
        self.branch_1_seperable_conv = SeparableConv2d(num_channels_in,
                                                       num_channels_out,
                                                       kernel_size=(3, 3),
                                                       padding="valid")
        self.branch_1_batch_norm = nn.BatchNorm2d(num_channels_out)
        self.branch_1_leaky_relu = nn.LeakyReLU()

        # Branch 2
        self.branch_2_reflection_padding = nn.ReflectionPad2d((2, 2))
        self.branch_2_seperable_conv = SeparableConv2d(num_channels_in,
                                                       num_channels_out,
                                                       kernel_size=(3, 3),
                                                       padding="valid")  # TODO: Dialation rate....
        self.branch_2_batch_norm = nn.BatchNorm2d(num_channels_out)
        self.branch_2_leaky_relu = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

