import torch


class TorchTransformWrapper(torch.nn.Module):
    def __init__(self, transform: torch.nn.Module, max_pixel_value: float = 1) -> None:
        super().__init__()
        self._transform = transform
        self._max_pixel_value = max_pixel_value

    def __call__(self, x: torch.Tensor, apply_mask: bool) -> torch.Tensor:
        """
        Calls the wrapped transform
        
        :param x: A torch tensor of [BATCH_SIZE x C + 1 x H x W] where C is the number of
        channels in the image, and the plus one is the mask. The mask is assumed to be the
        last channel.
        NOTE - BATCH_SIZE optional
        :param bool: Bool indicating if we should apply the mask after the transforms
        """
        batched = True
        if len(x.shape) == 3:
            batched = False
            x = x.unsqueeze(0)
        
        # Initialize the resultant tensor to an empty tensor the same shape  
        resultant = torch.empty_like(x)
        # Apply the transform
        resultant[:, 0:-1, :, :] = self._transform(x[:, 0:-1, :, :] / self._max_pixel_value)  * self._max_pixel_value
        # Clip back to 0 -> self._max_pixel_value
        resultant[:, 0:-1, :, :] = torch.clip(resultant[:, 0:-1, :, :], 0, self._max_pixel_value)
        # Copy the mask back over
        resultant[:, -1, :, :] = x[:, -1, :, :]
        # Apply the mask if the user wanted to remove superfulous values
        if apply_mask:
            mask = (x[:, -1, :, :] == 0)
            mask = torch.stack((mask,) * 3, dim=1)
            resultant[:, 0:-1, :, :][mask] = x[:, 0:-1, :, :][mask]

        if not batched:
            resultant = resultant.squeeze(0)

        return resultant
