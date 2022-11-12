import torch
import torchvision.transforms as T
from torchvision.transforms import functional as F

from typing import Union, List


def horizontal_flip(x: torch.Tensor, horizontal_dim: int = 0) -> torch.Tensor:
    """
    Takes an image and horizontally flips it
    :param x: Tensor to flip, assumed to be Height x Width x Channel unless
    horizontal_dim is set to something other than 0
    :param horizontal_dim: Which dimension is the horizontal dimension
    :return: Flipped Tensor
    """
    return torch.flip(x, [horizontal_dim,])


def gaussian_blur(x: torch.Tensor,
                  kernel_size: Union[(List, int)] = (5, 5),
                  sigma: List = (0.5, 0.5),
                  green_threshold: int = 10) -> torch.Tensor:
    """
    Applies a gaussian blur to an image
    :param x: Tensor to apply blur to, assumed to be Height x Width x Channel
    :param kernel_size:
    :param sigma:
    :param green_threshold:
    :return:
    """
    mask = x[:, :, 2] < green_threshold
    gauss_blur = T.GaussianBlur(kernel_size, sigma)
    resultant = gauss_blur(x.permute(2, 0, 1)).permute(1, 2, 0)
    resultant[mask] = x[mask]
    return resultant


def jitter(x: torch.Tensor,
           factor: float = 255,
           brightness: float = None,
           contrast: float = None,
           saturation: float = None,
           hue: float = None,
           green_threshold: int = 10) -> torch.Tensor:
    """
    Adds color jitter to an image
    :param x:
    :param factor:
    :param brightness:
    :param contrast:
    :param saturation:
    :param hue:
    :return:
    """
    mask = x[:, :, 2] < green_threshold
    resultant = torch.Tensor(x).permute(2, 0, 1) / factor

    if brightness:
        assert (brightness > 0), "Please enter a brightness greater than 0"
        resultant = F.adjust_brightness(resultant, brightness)

    if contrast:
        assert (contrast > 0), "Please enter a contrast greater than 0"
        resultant = F.adjust_contrast(resultant, contrast)

    if saturation:
        assert (saturation > 0), "Please enter a saturation greater than 0"
        resultant = F.adjust_saturation(resultant, saturation)

    if hue:
        assert (-0.5 <= hue <= 0.5), "Please enter a hue between -0.5 and 0.5"
        resultant = F.adjust_hue(resultant, hue)

    resultant = resultant.permute(1, 2, 0) * factor
    resultant[mask] = x[mask]

    return resultant


def solarization(x: torch.Tensor,
                 factor: float = 255,
                 threshold: float = 0.5,
                 green_threshold: int = 10) -> torch.Tensor:
    """
    Applies solarization to an image
    :param x:
    :param factor:
    :param threshold:
    :param green_threshold:
    :return:
    """
    assert (0 < threshold < 1), "Please use a threshold between 0 and 1"
    mask = x[:, :, 2] < green_threshold
    resultant = F.solarize(x.permute(2, 0, 1) / factor, threshold).permute(1, 2, 0) * factor
    resultant[mask] = x[mask]
    return resultant


def grayscale(x: torch.Tensor) -> torch.Tensor:
    """
    Converts an image to grayscale
    :param x:
    :return:
    """
    return F.rgb_to_grayscale(x.permute(2, 0, 1), num_output_channels=3).permute(1, 2, 0)


def rotate(x: torch.Tensor, angle: float = 90) -> torch.Tensor:
    """
    Rotates the image by the desired angle
    :param x:
    :param angle:
    :return:
    """
    return F.rotate(x.permute(2, 0, 1), angle=angle).permute(1, 2, 0)
