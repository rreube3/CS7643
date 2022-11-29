import torch
import torchvision.transforms as T
from torchvision.transforms import functional as F

from typing import Union, List


class EyeDatasetCustomTransform(torch.nn.Module):
    def __init__(self, max_pixel_value: float = 255, mask_threshold: float = 0.5):
        super().__init__()
        self._max_pixel_value = max_pixel_value
        self._mask_threshold = mask_threshold

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: [BATCH_SIZE x H x W x C]
        """
        batched = True
        if len(x.shape) == 3:
            batched = False
            x = x.unsqueeze(0)

        # don't need this
        # x = x.permute([0, 3, 1, 2])
        x /= self._max_pixel_value
        x[:, -1, :, :] = (x[:, -1, :, :] >= self._mask_threshold).float()

        if not batched:
            x = x.squeeze(0)

        return x
    