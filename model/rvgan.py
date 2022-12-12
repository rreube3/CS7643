import torch
from torch import nn
from torchvision.transforms import Resize
from typing import Union, Tuple, List, Dict


"""REF: https://arxiv.org/pdf/2101.00535.pdf"""
"""TF/KERAS Reference Implementation: https://github.com/SharifAmit/RVGAN"""


class SeparableConv2d(nn.Module):
    """REF: https://arxiv.org/pdf/1706.03059.pdf"""
    """REF: https://stackoverflow.com/questions/65154182/implement-separableconv2d-in-pytorch"""
    def __init__(self,
                 num_channels_in: int,
                 num_channels_out: int,
                 kernel_size: Union[int, Tuple],
                 padding: Union[str, Tuple] = "same",
                 dilation: int = 1):
        super().__init__()
        self.depthwise_conv = nn.Conv2d(num_channels_in,
                                        num_channels_in,
                                        kernel_size=kernel_size,
                                        groups=num_channels_in,
                                        bias=False,
                                        padding=padding,
                                        dilation=dilation)
        self.pointwise_conv = nn.Conv2d(num_channels_in,
                                        num_channels_out,
                                        kernel_size=1,
                                        bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


class SpatialFeatureAgregation(nn.Module):
    def __init__(self,
                 num_channels_in: int,
                 num_channels_out: int):
        super().__init__()

        self.seperable_conv = SeparableConv2d(num_channels_in,
                                              num_channels_out,
                                              kernel_size=(3, 3),
                                              padding="same")
        self.batch_norm_1 = nn.BatchNorm2d(num_channels_out)
        self.leaky_relu_1 = nn.LeakyReLU()
        
        self.conv = nn.Conv2d(num_channels_out,
                              num_channels_out,
                              kernel_size=(3, 3),
                              padding="same")
        self.batch_norm_2 = nn.BatchNorm2d(num_channels_out)
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
        super().__init__()
        
        self.reflection_padding = nn.ReflectionPad2d(padding=1)
        self.seperable_conv = SeparableConv2d(num_channels_in,
                                              num_channels_out,
                                              kernel_size=(3, 3),
                                              padding="valid")
        self.batch_norm = nn.BatchNorm2d(num_channels_out)
        self.leaky_relu = nn.LeakyReLU()

        # Branch 1
        self.branch_1_reflection_padding = nn.ReflectionPad2d(padding=1)
        self.branch_1_seperable_conv = SeparableConv2d(num_channels_out,
                                                       num_channels_out,
                                                       kernel_size=(3, 3),
                                                       padding="valid")
        self.branch_1_batch_norm = nn.BatchNorm2d(num_channels_out)
        self.branch_1_leaky_relu = nn.LeakyReLU()

        # Branch 2
        self.branch_2_reflection_padding = nn.ReflectionPad2d(padding=2)
        self.branch_2_seperable_conv = SeparableConv2d(num_channels_out,
                                                       num_channels_out,
                                                       kernel_size=(3, 3),
                                                       padding="valid",
                                                       dilation=2)
        self.branch_2_batch_norm = nn.BatchNorm2d(num_channels_out)
        self.branch_2_leaky_relu = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.reflection_padding(x)
        x = self.seperable_conv(x)
        x = self.batch_norm(x)
        x = self.leaky_relu(x)

        # Branch 1
        branch_1_x = self.branch_1_reflection_padding(x)
        branch_1_x = self.branch_1_seperable_conv(branch_1_x)
        branch_1_x = self.branch_1_batch_norm(branch_1_x)
        branch_1_x = self.branch_1_leaky_relu(branch_1_x)

        # Branch 2
        branch_2_x = self.branch_2_reflection_padding(x)
        branch_2_x = self.branch_2_seperable_conv(branch_2_x)
        branch_2_x = self.branch_2_batch_norm(branch_2_x)
        branch_2_x = self.branch_2_leaky_relu(branch_2_x)

        # Combine the stem and two branches
        x = torch.add(torch.add(x, branch_1_x), branch_2_x)

        return x


class DiscriminatorResidualBlock(nn.Module):
    def __init__(self,
                 num_channels_in: int,
                 num_channels_out: int):
        super().__init__()

        # Branch 1
        self.branch_1_reflection_padding = nn.ReflectionPad2d(padding=1)
        self.branch_1_conv = nn.Conv2d(num_channels_in,
                                       num_channels_out,
                                       kernel_size=(2, 2),
                                       dilation=2,
                                       padding="valid")
        self.branch_1_batch_norm = nn.BatchNorm2d(num_channels_out)
        self.branch_1_leaky_relu = nn.LeakyReLU()

        # Branch 2
        self.branch_2_reflection_padding = nn.ReflectionPad2d(padding=1)
        self.branch_2_seperable_conv = SeparableConv2d(num_channels_in,
                                                       num_channels_out,
                                                       kernel_size=(2, 2),
                                                       padding="valid",
                                                       dilation=2)
        self.branch_2_batch_norm = nn.BatchNorm2d(num_channels_out)
        self.branch_2_leaky_relu = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Branch 1
        branch_1_x = self.branch_1_reflection_padding(x)
        branch_1_x = self.branch_1_conv(branch_1_x)
        branch_1_x = self.branch_1_batch_norm(branch_1_x)
        branch_1_x = self.branch_1_leaky_relu(branch_1_x)

        # Branch 2
        branch_2_x = self.branch_2_reflection_padding(x)
        branch_2_x = self.branch_2_seperable_conv(branch_2_x)
        branch_2_x = self.branch_2_batch_norm(branch_2_x)
        branch_2_x = self.branch_2_leaky_relu(branch_2_x)

        # Combine the stem and two branches
        x = torch.add(branch_1_x, branch_2_x)

        return x


class Generator(nn.Module):
    def __init__(self,
                 num_channels_in: int,
                 num_channels_out: int = 1,
                 init_num_channels: int = 128,
                 num_up_and_down_sampling_layers: int = 2,
                 num_residual_blocks: int = 9):
        super().__init__()

        self.reflection_padding_1 = nn.ReflectionPad2d(padding=3)
        self.conv_1 = nn.Conv2d(num_channels_in,
                                init_num_channels,
                                kernel_size=(7, 7),
                                padding="valid")
        self.batch_1 = nn.BatchNorm2d(init_num_channels)
        self.leaky_1 = nn.LeakyReLU()

        # Downsampling layers (encoder)
        self.encoders = nn.ModuleList()
        # SFA layers
        self.sfa_layers = nn.ModuleList()
        # Upsampling layers (decoder)
        self.decoders = nn.ModuleList()
        
        for i in range(num_up_and_down_sampling_layers):
            num_channels_in_encoder = 2**i * init_num_channels
            num_channels_out_encoder = 2**(i + 1) * init_num_channels
            self.encoders.append(EncoderBlock(num_channels_in_encoder,
                                              num_channels_out_encoder))
            self.sfa_layers.append(SpatialFeatureAgregation(num_channels_in_encoder,
                                                            num_channels_in_encoder))
            self.decoders.append(DecoderBlock(num_channels_out_encoder,
                                              num_channels_in_encoder))

        # Residual blocks
        self.residual_blocks = nn.ModuleList()
        num_residual_filters = init_num_channels * 2**num_up_and_down_sampling_layers
        for i in range(num_residual_blocks):
            self.residual_blocks.append(GeneratorResidualBlock(num_residual_filters,
                                                               num_residual_filters))


        # Model head
        self.reflection_padding_2 = nn.ReflectionPad2d(padding=3)
        self.conv_2 = nn.Conv2d(init_num_channels,
                                num_channels_out,
                                kernel_size=(7, 7),
                                padding="valid")
        self.activation = nn.Sigmoid()

    def forward(self,
                x: torch.Tensor,
                residual_skip_connection: torch.Tensor = None,
                apply_activation: bool = True) -> torch.Tensor:
        x_head = self.leaky_1(self.batch_1(self.conv_1(self.reflection_padding_1(x))))
        
        # The output x_head is one of the skip connections
        skip_connections = [x_head, ]
        encoder_outputs = []
        # Create the encoders
        cur_x = x_head
        for i, encoder in enumerate(self.encoders):
            cur_x = encoder(cur_x)
            encoder_outputs.append(cur_x)
            if i < len(self.encoders) - 1:
                skip_connections.append(cur_x)
        # Create the SFA ouputs
        sfa_outputs = []
        for skip_connection, sfa in zip(skip_connections, self.sfa_layers):
            sfa_outputs.append(sfa(skip_connection))
        # If a residual skip connection is passed in, add it to the resids
        if residual_skip_connection is not None:
            cur_x = cur_x + residual_skip_connection
        # Run the residual blocks
        for resid_block in self.residual_blocks:
            cur_x = resid_block(cur_x)
        # Run up through the decoders / skip connections
        for decoder, skip_connection in zip(self.decoders[::-1], skip_connections[::-1]):
            cur_x = decoder(cur_x)
            cur_x = cur_x + skip_connection
        # Run through the head
        cur_x = self.conv_2(self.reflection_padding_2(cur_x))

        if apply_activation:
            cur_x = self.activation(cur_x)

        return cur_x


class CoarseGenerator(Generator):
    def __init__(self,
                 num_channels_in: int,
                 num_channels_out: int = 1,
                 init_num_channels: int = 128,
                 num_up_and_down_sampling_layers: int = 2,
                 num_residual_blocks: int = 9):
        super().__init__(num_channels_in,
                         num_channels_out,
                         init_num_channels,
                         num_up_and_down_sampling_layers,
                         num_residual_blocks)


class FineGenerator(Generator):
    def __init__(self,
                 num_channels_in: int,
                 num_channels_out: int = 1,
                 init_num_channels: int = 128,
                 num_up_and_down_sampling_layers: int = 1,
                 num_residual_blocks: int = 3):
        super().__init__(num_channels_in,
                         num_channels_out,
                         init_num_channels,
                         num_up_and_down_sampling_layers,
                         num_residual_blocks)


class Discriminator(nn.Module):
    def __init__(self,
                 num_channels_in_image: int,
                 num_channels_in_label: int = 1,
                 num_filters: int = 64,
                 num_layers: int = 3):
        super().__init__()

        # Create the downsample filters
        num_channels_in = num_channels_in_image + num_channels_in_label
        self.encoders = nn.ModuleList()
        self.residual_layers = nn.ModuleList()
        for _ in range(num_layers):
            # Downsample
            self.encoders.append(EncoderBlock(num_channels_in,
                                              num_filters))
            num_channels_in = num_filters
            # Residual
            self.residual_layers.append(DiscriminatorResidualBlock(num_filters,
                                                                   num_filters))

        # Create the upsample filters
        self.decoders = nn.ModuleList()
        for _ in range(num_layers):
            self.decoders.append(DecoderBlock(num_filters,
                                              num_filters))

        # Discriminator head
        self.conv = nn.Conv2d(num_filters,
                              num_channels_in_label,
                              kernel_size=(4, 4),
                              padding="same")
        self.activation = nn.Tanh()

    def forward(self,
                x: torch.Tensor,
                vessel_labels: torch.Tensor) -> Tuple[torch.Tensor, List]:
        # Concatenate the image and vessel labels
        x = torch.concat([x, vessel_labels], dim=1)

        # Initialize the list of features
        features = []
        
        # Run through the downsample layers appending each downsample to the feature list
        for (encoder, residual_layer) in zip(self.encoders, self.residual_layers):
            x = encoder(x)
            x = residual_layer(x)
            features.append(x)

        # Run through the upsample layers appending each downsample to the feature list
        for decoder in self.decoders:
            x = decoder(x)
            features.append(x)

        # Run through the head
        x = self.conv(x)
        x = self.activation(x)

        return x, features


class CoarseAndFineGenerators(nn.Module):
    def __init__(
        self,
        num_channels_in: int,
        num_channels_out: int = 1,
        init_num_channels: int = 128,
        num_up_and_down_sampling_layers_fine: int = 1,
        num_residual_blocks_fine: int = 3,
        num_up_and_down_sampling_layers_coarse: int = 2,
        num_residual_blocks_coarse: int = 9
    ):
        super().__init__()

        self.fine_generator = FineGenerator(
            num_channels_in,
            num_channels_out,
            init_num_channels,
            num_up_and_down_sampling_layers_fine,
            num_residual_blocks_fine
        )

        self.coarse_generator = CoarseGenerator(
            num_channels_in,
            num_channels_out,
            init_num_channels,
            num_up_and_down_sampling_layers_coarse,
            num_residual_blocks_coarse
        )
        
        # Compute the decimination factor
        self.decimination_factor = 2**(num_up_and_down_sampling_layers_coarse - 1)

    def forward(self,
                x: torch.Tensor,
                decimated_x: torch.Tensor = None,
                apply_activation: bool = True
        ) -> Tuple[torch.Tensor, torch.Tensor]:

        # Create the decimated image for the coarse
        if decimated_x is None:
            decimated_shape = [int(d / self.decimination_factor) for d in x.shape[2:]]
            resizer = Resize(size=decimated_shape)
            decimated_x = resizer(x)

        # Run the generators
        coarse_generator_out = self.coarse_generator(decimated_x, apply_activation=apply_activation)
        if apply_activation:
            fine_generator_out = self.fine_generator(x,
                                                     residual_skip_connection=coarse_generator_out)
        else:
            fine_generator_out = self.fine_generator(x,
                                                     residual_skip_connection=self.coarse_generator.activation(coarse_generator_out))
        return coarse_generator_out, fine_generator_out


class RVGAN(nn.Module):
    def __init__(self,
                 num_channels_in: int,
                 num_coarse_up_and_down_sampling_layers: int = 2):
        super().__init__()
        # Create the generators
        self.generators = CoarseAndFineGenerators(num_channels_in)
        # Create the discriminators
        self.coarse_discriminator = Discriminator(num_channels_in)
        self.fine_discriminator = Discriminator(num_channels_in)
        # Compute the decimination factor
        self.decimination_factor = 2**(num_coarse_up_and_down_sampling_layers - 1)

    def forward(self,
                x: torch.Tensor,
                vessel_labels: torch.Tensor = None) -> Dict:
        # Create the decimated image for the coarse
        decimated_shape = [int(d / self.decimination_factor) for d in x.shape[2:]]
        resizer = Resize(size=decimated_shape)
        decimated_x = resizer(x)
        if vessel_labels is not None:
            decimated_vessel_lbls = resizer(vessel_labels)
        else:
            decimated_vessel_lbls = None

        # Run the GANs
        coarse_generator_out, fine_generator_out = self.generators(
            x, decimated_x=decimated_x
        )

        # Run the Discriminators sending the real vessel map and the GAN generated one
        if vessel_labels is not None:
            real_coarse_discriminator_out, real_coarse_discriminator_features = self.coarse_discriminator(
                decimated_x,
                vessel_labels=decimated_vessel_lbls
            )
            real_fine_discriminator_out, real_fine_discriminator_features = self.fine_discriminator(
                x,
                vessel_labels=vessel_labels
            )
        else:
            real_coarse_discriminator_out = None
            real_coarse_discriminator_features = None
            real_fine_discriminator_out = None
            real_fine_discriminator_features = None

        fake_coarse_discriminator_out, fake_coarse_discriminator_features = self.coarse_discriminator(
            decimated_x,
            vessel_labels=coarse_generator_out
        )
        fake_fine_discriminator_out, fake_fine_discriminator_features = self.fine_discriminator(
            x,
            vessel_labels=fine_generator_out
        )

        return {
            "Coarse Generator Out": coarse_generator_out,
            "Fine Generator Out": fine_generator_out,
            "Real": {
                "Coarse Discriminator Out": real_coarse_discriminator_out,
                "Coarse Discriminator Features": real_coarse_discriminator_features,
                "Fine Discriminator Out": real_fine_discriminator_out,
                "Fine Discriminator Features": real_fine_discriminator_features
            },
            "Fake": {
                "Coarse Discriminator Out": fake_coarse_discriminator_out,
                "Coarse Discriminator Features": fake_coarse_discriminator_features,
                "Fine Discriminator Out": fake_fine_discriminator_out,
                "Fine Discriminator Features": fake_fine_discriminator_features
            },
            "Vessel Labels": {
                "Fine": vessel_labels,
                "Coarse": decimated_vessel_lbls
            }
        }
