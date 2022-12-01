import torch
import torchvision.transforms as T
from torchvision.transforms import functional as F

from typing import Union, List


class EyeDatasetCustomTransform(torch.nn.Module):
    def __init__(self, max_pixel_value: float = 1, mask_threshold: float = 0.5):
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
        #from IPython.core.debugger import set_trace; set_trace()
        x /= self._max_pixel_value
        x[:, -1, :, :] = (x[:, -1, :, :] > self._mask_threshold).float()

        if not batched:
            x = x.squeeze(0)

        return x
    

class EyeMaskCustomTransform(torch.nn.Module):
    def __init__(self, pixel_count_threshold: int = 0, max_pixel_value: float = 1, mask_threshold: float = 0.5):
        super().__init__()
        self._pixel_count_threshold = pixel_count_threshold
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
            
        # Preserve the mask
        mask = x[:, -1, :, :] < self._mask_threshold
        # Take only the first channel
        x = x[:, 0, :, :]
        # Turn this into an Tensor of 0's and 1's
        x[x > self._pixel_count_threshold] = 1.0
        x[x < 1] = 0
        # Make sure masked values are 0
        x[mask] = 0
        # Make sure we are [BATCH_SIZE x C x H x W]
        x = x.unsqueeze(1)

        if not batched:
            x = x.squeeze(0)

        return x
    