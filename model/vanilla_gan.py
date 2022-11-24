import torch
import torch.nn as nn

#  Created by referencing:
#  Goodfellow, I. J., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., … & Bengio, Y. (2014). Generative adversarial networks. arXiv preprint arXiv:1406.2661.
#  MachineCurve. (2021, July 15). Creating DCGAN with PyTorch. https://www.machinecurve.com/index.php/2021/07/15/creating-dcgan-with-pytorch/
#  https://github.com/christianversloot/machine-learning-articles/blob/main/building-a-simple-vanilla-gan-with-pytorch.md

class Generator(nn.Module):
  """
    Vanilla GAN Generator - replace convolutional layers with linear layers for vanilla gan
    input_shape: input shape of noisy images
    output_shape: final image dimensions (multiplied channels*width*height)
  """
  def __init__(self,
               input_shape: int,
               output_shape: int):
    super().__init__()
    self.combine_layers = nn.Sequential(
      # layer 1 - linear, normalization, leaky ReLU
      nn.Linear(input_shape, 256, bias=False),
      nn.BatchNorm2d(256),
      nn.LeakyReLU(0.25),
      # layer 2 - linear, normalization, leaky ReLU
      nn.Linear(256, 512, bias=False),
      nn.BatchNorm2d(512),
      nn.LeakyReLU(0.25),
      # layer 3 - linear, normalization, leaky ReLU
      nn.Linear(512, 1024, bias=False),
      nn.BatchNorm2d(1024),
      nn.LeakyReLU(0.25),
      # layer 4 - linear, tanh
      nn.Linear(1024, output_shape, bias=False),
      nn.Tanh()
    )

  def forward(self, x):
    """Generator forward pass"""
    return self.combine_layers(x)


class Discriminator(nn.Module):
  """
    Vanilla GAN Discriminator - replace convolutional layers with linear layers for vanilla gan
    output_shape: final image dimensions (multiplied channels*width*height) passed for classification
  """
  def __init__(self,
               output_shape: int):
    super().__init__()
    self.combine_layers = nn.Sequential(
      nn.Linear(output_shape, 1024),
      nn.LeakyReLU(0.25),
      nn.Linear(1024, 512),
      nn.LeakyReLU(0.25),
      nn.Linear(512, 256),
      nn.LeakyReLU(0.25),
      nn.Linear(256, 1),
      nn.Sigmoid()
    )

  def forward(self, x):
    """Discriminator forward pass"""
    return self.combine_layers(x)