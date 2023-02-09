import warnings
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks import Convolution
from monai.networks.layers.utils import get_act_layer
from monai.utils import LossReduction
from monai.utils.enums import StrEnum
from torch.nn.modules.loss import _Loss


class AdversarialCriterions(StrEnum):
    BCE = "bce"
    HINGE = "hinge"
    LEAST_SQUARE = "least_squares"


class PatchAdversarialLoss(_Loss):
    def __init__(
        self,
        reduction: Union[LossReduction, str] = LossReduction.MEAN,
        criterion: str = AdversarialCriterions.LEAST_SQUARE.value,
    ) -> None:
        super().__init__(reduction=LossReduction(reduction).value)
        self.real_label = 1.0
        self.fake_label = 0.0
        if criterion == AdversarialCriterions.BCE.value:
            self.activation = get_act_layer("SIGMOID")
            self.loss_fct = torch.nn.BCELoss(reduction=reduction)
        elif criterion == AdversarialCriterions.HINGE.value:
            self.activation = get_act_layer("TANH")
            self.fake_label = -1.0
        elif criterion == AdversarialCriterions.LEAST_SQUARE.value:
            self.activation = get_act_layer(name=("LEAKYRELU", {"negative_slope": 0.05}))
            self.loss_fct = torch.nn.MSELoss(reduction=reduction)

        self.criterion = criterion
        self.reduction = reduction

    def get_target_tensor(self, input: torch.FloatTensor, target_is_real: bool) -> torch.Tensor:
        filling_label = self.real_label if target_is_real else self.fake_label
        label_tensor = torch.tensor(1).fill_(filling_label).type(input.type())
        label_tensor.requires_grad_(False)
        return label_tensor.expand_as(input)

    def get_zero_tensor(self, input: torch.FloatTensor) -> torch.Tensor:
        zero_label_tensor = torch.tensor(0).type(input[0].type())
        zero_label_tensor.requires_grad_(False)
        return zero_label_tensor.expand_as(input)

    def forward(
        self, input: Union[torch.FloatTensor, list], target_is_real: bool, for_discriminator: bool
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        if not for_discriminator and not target_is_real:
            target_is_real = True  # With generator, we always want this to be true!
            warnings.warn(
                "Variable target_is_real has been set to False, but for_discriminator is set"
                "to False. To optimise a generator, target_is_real must be set to True."
            )

        if type(input) is not list:
            input = [input]
        target_ = []
        for disc_ind, disc_out in enumerate(input):
            if self.criterion != AdversarialCriterions.HINGE.value:
                target_.append(self.get_target_tensor(disc_out, target_is_real))
            else:
                target_.append(self.get_zero_tensor(disc_out))

        # Loss calculation
        loss = []
        for disc_ind, disc_out in enumerate(input):
            disc_out = self.activation(disc_out)
            if self.criterion == AdversarialCriterions.HINGE.value and not target_is_real:
                loss_ = self.forward_single(-disc_out, target_[disc_ind])
            else:
                loss_ = self.forward_single(disc_out, target_[disc_ind])
            loss.append(loss_)

        if loss is not None:
            if self.reduction == LossReduction.MEAN.value:
                loss = torch.mean(torch.stack(loss))
            elif self.reduction == LossReduction.SUM.value:
                loss = torch.sum(torch.stack(loss))

        return loss

    def forward_single(self, input: torch.FloatTensor, target: torch.FloatTensor) -> Optional[torch.Tensor]:
        if (
            self.criterion == AdversarialCriterions.BCE.value
            or self.criterion == AdversarialCriterions.LEAST_SQUARE.value
        ):
            return self.loss_fct(input, target)
        elif self.criterion == AdversarialCriterions.HINGE.value:
            minval = torch.min(input - 1, self.get_zero_tensor(input))
            return -torch.mean(minval)
        else:
            return None


def kl_loss_function(z_mu, z_sigma):
    kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3, 4])
    return torch.sum(kl_loss) / kl_loss.shape[0]


class Upsample(nn.Module):
    """
    Convolution-based upsampling layer.

    Args:
        spatial_dims: number of spatial dimensions (1D, 2D, 3D).
        in_channels: number of input channels to the layer.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
    ) -> None:
        super().__init__()
        self.conv = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=in_channels,
            strides=1,
            kernel_size=3,
            padding=1,
            conv_only=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    Convolution-based downsampling layer.

    Args:
        spatial_dims: number of spatial dimensions (1D, 2D, 3D).
        in_channels: number of input channels.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
    ) -> None:
        super().__init__()
        self.pad = (0, 1) * spatial_dims

        self.conv = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=in_channels,
            strides=2,
            kernel_size=3,
            padding=0,
            conv_only=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.pad(x, self.pad, mode="constant", value=0.0)
        x = self.conv(x)
        return x


class ResBlock(nn.Module):
    """
    Residual block consisting of a cascade of 2 convolutions + activation + normalisation block, and a
    residual connection between input and output.

    Args:
        spatial_dims: number of spatial dimensions (1D, 2D, 3D).
        in_channels: input channels to the layer.
        norm_num_groups: number of groups involved for the group normalisation layer. Ensure that your number of
            channels is divisible by this number.
        norm_eps: epsilon for the normalisation.
        out_channels: number of output channels.
    """

    def __init__(
        self, spatial_dims: int, in_channels: int, norm_num_groups: int, norm_eps: float, out_channels: int
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels

        self.norm1 = nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=norm_eps, affine=True)
        self.conv1 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            strides=1,
            kernel_size=3,
            padding=1,
            conv_only=True,
        )
        self.norm2 = nn.GroupNorm(num_groups=norm_num_groups, num_channels=out_channels, eps=norm_eps, affine=True)
        self.conv2 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            strides=1,
            kernel_size=3,
            padding=1,
            conv_only=True,
        )

        if self.in_channels != self.out_channels:
            self.nin_shortcut = Convolution(
                spatial_dims=spatial_dims,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                strides=1,
                kernel_size=1,
                padding=0,
                conv_only=True,
            )
        else:
            self.nin_shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    """
    Attention block.

    Args:
        spatial_dims: number of spatial dimensions (1D, 2D, 3D).
        in_channels: number of input channels.
        norm_num_groups: number of groups involved for the group normalisation layer. Ensure that your number of
            channels is divisible by this number.
        norm_eps: epsilon for the normalisation.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        norm_num_groups: int,
        norm_eps: float,
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels

        self.norm = nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=norm_eps, affine=True)
        self.q = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=in_channels,
            strides=1,
            kernel_size=1,
            padding=0,
            conv_only=True,
        )
        self.k = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=in_channels,
            strides=1,
            kernel_size=1,
            padding=0,
            conv_only=True,
        )
        self.v = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=in_channels,
            strides=1,
            kernel_size=1,
            padding=0,
            conv_only=True,
        )
        self.proj_out = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=in_channels,
            strides=1,
            kernel_size=1,
            padding=0,
            conv_only=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # Compute attention
        b = q.shape[0]
        c = q.shape[1]
        h = q.shape[2]
        w = q.shape[3]
        # in order to Torchscript work, we initialise d = 1
        d = 1

        if self.spatial_dims == 3:
            d = q.shape[4]
        n_spatial_elements = h * w * d

        q = q.reshape(b, c, n_spatial_elements)
        q = q.permute(0, 2, 1)
        k = k.reshape(b, c, n_spatial_elements)
        w_ = torch.bmm(q, k)
        w_ = w_ * (int(c) ** (-0.5))
        w_ = F.softmax(w_, dim=2)

        # Attend to values
        v = v.reshape(b, c, n_spatial_elements)
        w_ = w_.permute(0, 2, 1)
        h_ = torch.bmm(v, w_)

        if self.spatial_dims == 2:
            h_ = h_.reshape(b, c, h, w)
        if self.spatial_dims == 3:
            h_ = h_.reshape(b, c, h, w, d)

        h_ = self.proj_out(h_)

        return x + h_


class Encoder(nn.Module):
    """
    Convolutional cascade that downsamples the image into a spatial latent space.

    Args:
        spatial_dims: number of spatial dimensions (1D, 2D, 3D).
        in_channels: number of input channels.
        num_channels: number of filters in the first downsampling.
        out_channels: number of channels in the bottom layer (latent space) of the autoencoder.
        ch_mult: list of multipliers of num_channels in the initial layer and in  each downsampling layer. Example: if
            you want three downsamplings, you have to input a 4-element list. If you input [1, 1, 2, 2],
            the first downsampling will leave num_channels to num_channels, the next will multiply num_channels by 2,
            and the next will multiply num_channels*2 by 2 again, resulting in 8, 8, 16 and 32 channels.
        num_res_blocks: number of residual blocks (see ResBlock) per level.
        norm_num_groups: number of groups for the GroupNorm layers, num_channels must be divisible by this number.
        norm_eps: epsilon for the normalization.
        attention_levels: indicate which level from ch_mult contain an attention block.
        with_nonlocal_attn: if True use non-local attention block.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        num_channels: int,
        out_channels: int,
        ch_mult: Sequence[int],
        num_res_blocks: int,
        norm_num_groups: int,
        norm_eps: float,
        attention_levels: Optional[Sequence[bool]] = None,
        with_nonlocal_attn: bool = True,
    ) -> None:
        super().__init__()

        if attention_levels is None:
            attention_levels = (False,) * len(ch_mult)

        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.num_channels = num_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.norm_num_groups = norm_num_groups
        self.norm_eps = norm_eps
        self.attention_levels = attention_levels

        in_ch_mult = (1,) + tuple(ch_mult)

        blocks = []
        # Initial convolution
        blocks.append(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=num_channels,
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
            )
        )

        # Residual and downsampling blocks
        for i in range(len(ch_mult)):
            block_in_ch = num_channels * in_ch_mult[i]
            block_out_ch = num_channels * ch_mult[i]
            for _ in range(self.num_res_blocks):
                blocks.append(
                    ResBlock(
                        spatial_dims=spatial_dims,
                        in_channels=block_in_ch,
                        norm_num_groups=norm_num_groups,
                        norm_eps=norm_eps,
                        out_channels=block_out_ch,
                    )
                )
                block_in_ch = block_out_ch
                if attention_levels[i]:
                    blocks.append(AttnBlock(spatial_dims, block_in_ch, norm_num_groups, norm_eps))

            if i != len(ch_mult) - 1:
                blocks.append(Downsample(spatial_dims, block_in_ch))

        # Non-local attention block
        if with_nonlocal_attn is True:
            blocks.append(ResBlock(spatial_dims, block_in_ch, norm_num_groups, norm_eps, block_in_ch))
            blocks.append(AttnBlock(spatial_dims, block_in_ch, norm_num_groups, norm_eps))
            blocks.append(ResBlock(spatial_dims, block_in_ch, norm_num_groups, norm_eps, block_in_ch))

        # Normalise and convert to latent size
        blocks.append(nn.GroupNorm(num_groups=norm_num_groups, num_channels=block_in_ch, eps=norm_eps, affine=True))
        blocks.append(
            Convolution(
                spatial_dims=self.spatial_dims,
                in_channels=block_in_ch,
                out_channels=out_channels,
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
            )
        )

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class Decoder(nn.Module):
    """
    Convolutional cascade upsampling from a spatial latent space into an image space.

    Args:
        spatial_dims: number of spatial dimensions (1D, 2D, 3D).
        num_channels: number of filters in the last upsampling.
        in_channels: number of channels in the bottom layer (latent space) of the autoencoder.
        out_channels: number of output channels.
        ch_mult: list of multipliers of num_channels that make for all the upsampling layers before the last. In the
            last layer, there will be a transition from num_channels to out_channels. In the layers before that,
            channels will be the product of the previous number of channels by ch_mult.
        num_res_blocks: number of residual blocks (see ResBlock) per level.
        norm_num_groups: number of groups for the GroupNorm layers, num_channels must be divisible by this number.
        norm_eps: epsilon for the normalization.
        attention_levels: indicate which level from ch_mult contain an attention block.
        with_nonlocal_attn: if True use non-local attention block.
    """

    def __init__(
        self,
        spatial_dims: int,
        num_channels: int,
        in_channels: int,
        out_channels: int,
        ch_mult: Sequence[int],
        num_res_blocks: int,
        norm_num_groups: int,
        norm_eps: float,
        attention_levels: Optional[Sequence[bool]] = None,
        with_nonlocal_attn: bool = True,
    ) -> None:
        super().__init__()

        if attention_levels is None:
            attention_levels = (False,) * len(ch_mult)

        self.spatial_dims = spatial_dims
        self.num_channels = num_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ch_mult = ch_mult
        self.num_res_blocks = num_res_blocks
        self.norm_num_groups = norm_num_groups
        self.norm_eps = norm_eps
        self.attention_levels = attention_levels

        block_in_ch = num_channels * self.ch_mult[-1]

        blocks = []
        # Initial convolution
        blocks.append(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=block_in_ch,
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
            )
        )

        # Non-local attention block
        if with_nonlocal_attn is True:
            blocks.append(ResBlock(spatial_dims, block_in_ch, norm_num_groups, norm_eps, block_in_ch))
            blocks.append(AttnBlock(spatial_dims, block_in_ch, norm_num_groups, norm_eps))
            blocks.append(ResBlock(spatial_dims, block_in_ch, norm_num_groups, norm_eps, block_in_ch))

        for i in reversed(range(len(ch_mult))):
            block_out_ch = num_channels * self.ch_mult[i]

            for _ in range(self.num_res_blocks):
                blocks.append(ResBlock(spatial_dims, block_in_ch, norm_num_groups, norm_eps, block_out_ch))
                block_in_ch = block_out_ch

                if attention_levels[i]:
                    blocks.append(AttnBlock(spatial_dims, block_in_ch, norm_num_groups, norm_eps))

            if i != 0:
                blocks.append(Upsample(spatial_dims, block_in_ch))

        blocks.append(nn.GroupNorm(num_groups=norm_num_groups, num_channels=block_in_ch, eps=norm_eps, affine=True))
        blocks.append(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=block_in_ch,
                out_channels=out_channels,
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
            )
        )

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class AutoencoderKL(nn.Module):
    """
    Autoencoder model with KL-regularized latent space based on
    Rombach et al. "High-Resolution Image Synthesis with Latent Diffusion Models" https://arxiv.org/abs/2112.10752
    and Pinaya et al. "Brain Imaging Generation with Latent Diffusion Models" https://arxiv.org/abs/2209.07162

    Args:
        spatial_dims: number of spatial dimensions (1D, 2D, 3D).
        in_channels: number of input channels.
        out_channels: number of output channels.
        num_channels: number of filters in the first downsampling / last upsampling.
        latent_channels: latent embedding dimension.
        ch_mult: multiplier of the number of channels in each downsampling layer (+ initial one). i.e.: If you want 3
            downsamplings, it should be a 4-element list.
        num_res_blocks: number of residual blocks (see ResBlock) per level.
        norm_num_groups: number of groups for the GroupNorm layers, num_channels must be divisible by this number.
        norm_eps: epsilon for the normalization.
        attention_levels: indicate which level from ch_mult contain an attention block.
        with_encoder_nonlocal_attn: if True use non-local attention block in the encoder.
        with_decoder_nonlocal_attn: if True use non-local attention block in the decoder.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        num_channels: int,
        latent_channels: int,
        ch_mult: Sequence[int],
        num_res_blocks: int,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        attention_levels: Optional[Sequence[bool]] = None,
        with_encoder_nonlocal_attn: bool = True,
        with_decoder_nonlocal_attn: bool = True,
    ) -> None:
        super().__init__()
        if attention_levels is None:
            attention_levels = (False,) * len(ch_mult)

        # The number of channels should be multiple of num_groups
        if (num_channels % norm_num_groups) != 0:
            raise ValueError("AutoencoderKL expects number of channels being multiple of number of groups")

        if len(ch_mult) != len(attention_levels):
            raise ValueError("AutoencoderKL expects ch_mult being same size of attention_levels")

        self.encoder = Encoder(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            num_channels=num_channels,
            out_channels=latent_channels,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            attention_levels=attention_levels,
            with_nonlocal_attn=with_encoder_nonlocal_attn,
        )
        self.decoder = Decoder(
            spatial_dims=spatial_dims,
            num_channels=num_channels,
            in_channels=latent_channels,
            out_channels=out_channels,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            attention_levels=attention_levels,
            with_nonlocal_attn=with_decoder_nonlocal_attn,
        )
        self.quant_conv_mu = Convolution(
            spatial_dims=spatial_dims,
            in_channels=latent_channels,
            out_channels=latent_channels,
            strides=1,
            kernel_size=1,
            padding=0,
            conv_only=True,
        )
        self.quant_conv_log_sigma = Convolution(
            spatial_dims=spatial_dims,
            in_channels=latent_channels,
            out_channels=latent_channels,
            strides=1,
            kernel_size=1,
            padding=0,
            conv_only=True,
        )
        self.post_quant_conv = Convolution(
            spatial_dims=spatial_dims,
            in_channels=latent_channels,
            out_channels=latent_channels,
            strides=1,
            kernel_size=1,
            padding=0,
            conv_only=True,
        )
        self.latent_channels = latent_channels

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forwards an image through the spatial encoder, obtaining the latent mean and sigma representations.

        Args:
            x: BxCx[SPATIAL DIMS] tensor

        """
        h = self.encoder(x)

        z_mu = self.quant_conv_mu(h)
        z_log_var = self.quant_conv_log_sigma(h)
        z_log_var = torch.clamp(z_log_var, -30.0, 20.0)
        z_sigma = torch.exp(z_log_var / 2)

        return z_mu, z_sigma

    def sampling(self, z_mu: torch.Tensor, z_sigma: torch.Tensor) -> torch.Tensor:
        """
        From the mean and sigma representations resulting of encoding an image through the latent space,
        obtains a noise sample resulting from sampling gaussian noise, multiplying by the variance (sigma) and
        adding the mean.

        Args:
            z_mu: Bx[Z_CHANNELS]x[LATENT SPACE SIZE] mean vector obtained by the encoder when you encode an image
            z_sigma: Bx[Z_CHANNELS]x[LATENT SPACE SIZE] variance vector obtained by the encoder when you encode an image

        Returns:
            sample of shape Bx[Z_CHANNELS]x[LATENT SPACE SIZE]
        """
        eps = torch.randn_like(z_sigma)
        z_vae = z_mu + eps * z_sigma
        return z_vae

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes and decodes an input image.

        Args:
            x: BxCx[SPATIAL DIMENSIONS] tensor.

        Returns:
            reconstructed image, of the same shape as input
        """
        z_mu, _ = self.encode(x)
        reconstruction = self.decode(z_mu)
        return reconstruction

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Based on a latent space sample, forwards it through the Decoder.

        Args:
            z: Bx[Z_CHANNELS]x[LATENT SPACE SHAPE]

        Returns:
            decoded image tensor
        """
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_mu, z_sigma = self.encode(x)
        z = self.sampling(z_mu, z_sigma)
        reconstruction = self.decode(z)
        return reconstruction, z_mu, z_sigma

    def encode_stage_2_inputs(self, x: torch.Tensor) -> torch.Tensor:
        z_mu, z_sigma = self.encode(x)
        z = self.sampling(z_mu, z_sigma)
        return z

    def decode_stage_2_outputs(self, z: torch.Tensor) -> torch.Tensor:
        image = self.decode(z)
        return image


class PatchDiscriminator(nn.Sequential):
    """
    Patch-GAN discriminator based on Pix2PixHD:
    High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs
    Ting-Chun Wang1, Ming-Yu Liu1, Jun-Yan Zhu2, Andrew Tao1, Jan Kautz1, Bryan Catanzaro (1)
    (1) NVIDIA Corporation, 2UC Berkeley
    In CVPR 2018.

    Args:
        num_layers_d: number of Convolution layers (Conv + activation + normalisation + [dropout]) in each
        of the discriminators. In each layer, the number of channels are doubled and the spatial size is
        divided by 2.
        spatial_dims: number of spatial dimensions (1D, 2D etc.)
        num_channels: number of filters in the first convolutional layer (double of the value is taken from then on)
        in_channels: number of input channels
        out_channels: number of output channels in each discriminator
        kernel_size: kernel size of the convolution layers
        activation: activation layer type
        norm: normalisation type
        bias: introduction of layer bias
        padding: padding to be applied to the convolutional layers
        dropout: proportion of dropout applied, defaults to 0.
        last_conv_kernel_size: kernel size of the last convolutional layer.
    """

    def __init__(
        self,
        num_layers_d: int,
        spatial_dims: int,
        num_channels: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        activation: Union[str, tuple] = "PRELU",
        norm: Union[str, tuple] = "INSTANCE",
        bias: bool = False,
        padding: Union[int, Sequence[int]] = 1,
        dropout: Union[float, tuple] = 0.0,
        last_conv_kernel_size: Optional[int] = None,
    ) -> None:

        super().__init__()
        self.num_layers_d = num_layers_d
        self.num_channels = num_channels
        if last_conv_kernel_size is None:
            last_conv_kernel_size = kernel_size

        self.add_module(
            "initial_conv",
            Convolution(
                spatial_dims=spatial_dims,
                kernel_size=kernel_size,
                in_channels=in_channels,
                out_channels=num_channels,
                act=activation,
                bias=True,
                norm=None,
                dropout=dropout,
                padding=padding,
                strides=2,
            ),
        )

        input_channels = num_channels
        output_channels = num_channels * 2

        # Initial Layer
        for l_ in range(self.num_layers_d):
            if l_ == self.num_layers_d - 1:
                stride = 1
            else:
                stride = 2
            layer = Convolution(
                spatial_dims=spatial_dims,
                kernel_size=kernel_size,
                in_channels=input_channels,
                out_channels=output_channels,
                act=activation,
                bias=bias,
                norm=norm,
                dropout=dropout,
                padding=padding,
                strides=stride,
            )
            self.add_module("%d" % l_, layer)
            input_channels = output_channels
            output_channels = output_channels * 2

        # Final layer
        self.add_module(
            "final_conv",
            Convolution(
                spatial_dims=spatial_dims,
                kernel_size=last_conv_kernel_size,
                in_channels=input_channels,
                out_channels=out_channels,
                bias=True,
                conv_only=True,
                padding=int((last_conv_kernel_size - 1) / 2),
                dropout=0.0,
                strides=1,
            ),
        )

        self.apply(self.initialise_weights)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """

        Args:
            x: input tensor
            feature-matching loss (regulariser loss) on the discriminators as well (see Pix2Pix paper).
        Returns:
            list of intermediate features, with the last element being the output.
        """
        out = [x]
        for submodel in self.children():
            intermediate_output = submodel(out[-1])
            out.append(intermediate_output)

        return out[1:]

    def initialise_weights(self, m: nn.Module) -> None:
        """
        Initialise weights of Convolution and BatchNorm layers.

        Args:
            m: instance of torch.nn.module (or of class inheriting torch.nn.module)
        """
        classname = m.__class__.__name__
        if classname.find("Conv2d") != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("Conv3d") != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("Conv1d") != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)


class SimpleNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.autoencoder = AutoencoderKL(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            num_channels=32,
            latent_channels=3,
            ch_mult=(1, 2, 2),
            num_res_blocks=1,
            norm_num_groups=16,
            attention_levels=(False, False, False),
            with_encoder_nonlocal_attn=False,
            with_decoder_nonlocal_attn=False,
        )
        self.discriminator = PatchDiscriminator(
            spatial_dims=3,
            num_layers_d=3,
            num_channels=32,
            in_channels=1,
            out_channels=1,
            kernel_size=4,
            activation="LEAKYRELU",
            norm="BATCH",
            bias=False,
            padding=1,
        )
