"""EDM Diffusion U-Net for world model, following DIAMOND architecture.

Key components:
  - U-Net encoder-decoder with skip connections
  - AdaGroupNorm for action + noise level conditioning
  - EDM (Karras et al.) preconditioning and noise schedule
  - Few-step sampling (3 Euler steps)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────── Building Blocks ────────────────────────────


class FourierFeatures(nn.Module):
    """Random Fourier features for encoding noise level sigma."""

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.register_buffer("freqs", torch.randn(dim // 2))

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        # sigma: (B,) -> (B, dim)
        x = sigma[:, None] * self.freqs[None, :] * 2 * math.pi
        return torch.cat([x.cos(), x.sin()], dim=-1)


class AdaGroupNorm(nn.Module):
    """Group norm with adaptive scale/shift from conditioning vector."""

    def __init__(self, channels: int, cond_channels: int, num_groups: int = 32):
        super().__init__()
        self.num_groups = min(num_groups, channels)
        self.norm = nn.GroupNorm(self.num_groups, channels, affine=False)
        self.proj = nn.Linear(cond_channels, channels * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        scale, shift = self.proj(cond)[:, :, None, None].chunk(2, dim=1)
        return x * (1 + scale) + shift


class ResBlock(nn.Module):
    """Residual block with two convolutions, each preceded by AdaGroupNorm."""

    def __init__(self, channels: int, cond_channels: int):
        super().__init__()
        self.norm1 = AdaGroupNorm(channels, cond_channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = AdaGroupNorm(channels, cond_channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        # Zero-init last conv for residual (better training start)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x, cond)))
        h = self.conv2(F.silu(self.norm2(h, cond)))
        return x + h


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class ChannelAdapter(nn.Module):
    """1x1 conv to match channel counts for skip connections."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ──────────────────────────────── U-Net ────────────────────────────────────


class UNet(nn.Module):
    """U-Net backbone for denoising diffusion.

    Input: (B, in_channels, H, W)
    Output: (B, out_channels, H, W)

    Conditioning via AdaGroupNorm at every ResBlock.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_channels: int = 256,
        channels: list = None,
        depths: list = None,
    ):
        super().__init__()
        if channels is None:
            channels = [64, 128, 128, 128]
        if depths is None:
            depths = [2, 2, 2, 2]

        num_levels = len(channels)
        self.conv_in = nn.Conv2d(in_channels, channels[0], 3, padding=1)

        # Encoder
        self.down_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        for i in range(num_levels):
            blocks = nn.ModuleList([ResBlock(channels[i], cond_channels) for _ in range(depths[i])])
            self.down_blocks.append(blocks)
            if i < num_levels - 1:
                self.downsamplers.append(Downsample(channels[i]))

        # Channel adapters for transitions between levels
        self.down_ch_adapt = nn.ModuleList()
        for i in range(num_levels - 1):
            self.down_ch_adapt.append(ChannelAdapter(channels[i], channels[i + 1]))

        # Middle
        self.mid_block1 = ResBlock(channels[-1], cond_channels)
        self.mid_block2 = ResBlock(channels[-1], cond_channels)

        # Decoder
        self.up_blocks = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        self.up_ch_adapt = nn.ModuleList()
        for i in reversed(range(num_levels)):
            # After concatenating skip connection, channels double
            in_ch = channels[i] * 2 if i < num_levels - 1 else channels[i]
            blocks = nn.ModuleList()
            for j in range(depths[i]):
                block_in = in_ch if j == 0 else channels[i]
                # Use ChannelAdapter if needed before ResBlock
                if block_in != channels[i]:
                    blocks.append(nn.Sequential(
                        ChannelAdapter(block_in, channels[i]),
                    ))
                blocks.append(ResBlock(channels[i], cond_channels))
            self.up_blocks.append(blocks)
            if i > 0:
                self.upsamplers.append(Upsample(channels[i]))
                self.up_ch_adapt.append(ChannelAdapter(channels[i], channels[i - 1]))

        self.norm_out = nn.GroupNorm(min(32, channels[0]), channels[0])
        self.conv_out = nn.Conv2d(channels[0], out_channels, 3, padding=1)
        # Zero-init output conv
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # Pad to multiple of 2^(num_levels-1)
        num_levels = len(self.down_blocks)
        factor = 2 ** (num_levels - 1)
        h, w = x.shape[2], x.shape[3]
        pad_h = (factor - h % factor) % factor
        pad_w = (factor - w % factor) % factor
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        x = self.conv_in(x)

        # Encoder
        skips = []
        for i in range(num_levels):
            for block in self.down_blocks[i]:
                x = block(x, cond)
            skips.append(x)
            if i < num_levels - 1:
                x = self.downsamplers[i](x)
                x = self.down_ch_adapt[i](x)

        # Middle
        x = self.mid_block1(x, cond)
        x = self.mid_block2(x, cond)

        # Decoder
        up_idx = 0
        for i in reversed(range(num_levels)):
            if i < num_levels - 1:
                x = self.upsamplers[up_idx - 1](x)
                x = self.up_ch_adapt[up_idx - 1](x)
                skip = skips[i]
                # Match sizes after upsampling
                if x.shape[2:] != skip.shape[2:]:
                    x = x[:, :, :skip.shape[2], :skip.shape[3]]
                x = torch.cat([x, skip], dim=1)

            for module in self.up_blocks[up_idx]:
                if isinstance(module, ResBlock):
                    x = module(x, cond)
                else:
                    x = module(x)
            up_idx += 1

        x = self.conv_out(F.silu(self.norm_out(x)))

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :h, :w]

        return x


# ──────────────────────────────── Denoiser ─────────────────────────────────


class Denoiser(nn.Module):
    """EDM denoiser wrapping the U-Net.

    Handles:
    - Noise level encoding (Fourier features)
    - Action embedding
    - EDM preconditioning (c_in, c_out, c_skip, c_noise)
    - Training (forward + loss)
    - Sampling (Euler ODE solver)
    """

    def __init__(
        self,
        img_channels: int = 3,
        img_size: int = 84,
        num_actions: int = 10,
        num_context_frames: int = 4,
        cond_channels: int = 256,
        channels: list = None,
        depths: list = None,
        sigma_data: float = 0.5,
        sigma_offset_noise: float = 0.3,
    ):
        super().__init__()
        self.img_channels = img_channels
        self.img_size = img_size
        self.num_actions = num_actions
        self.num_context = num_context_frames
        self.sigma_data = sigma_data
        self.sigma_offset_noise = sigma_offset_noise

        # Conditioning: Fourier(sigma) + action_emb -> MLP -> cond_channels
        fourier_dim = 64
        self.fourier = FourierFeatures(fourier_dim)
        self.action_emb = nn.Embedding(num_actions, cond_channels)
        self.cond_proj = nn.Sequential(
            nn.Linear(fourier_dim + cond_channels, cond_channels),
            nn.SiLU(),
            nn.Linear(cond_channels, cond_channels),
        )

        # U-Net: input = noisy_target (3ch) + context (L*3 ch)
        in_ch = img_channels * (1 + num_context_frames)
        self.unet = UNet(
            in_channels=in_ch,
            out_channels=img_channels,
            cond_channels=cond_channels,
            channels=channels,
            depths=depths,
        )

    def _edm_precond(self, sigma: torch.Tensor):
        """EDM preconditioning weights (Karras et al. Table 1, 'VP' row)."""
        s = self.sigma_data
        c_skip = s**2 / (sigma**2 + s**2)
        c_out = sigma * s / (sigma**2 + s**2).sqrt()
        c_in = 1 / (sigma**2 + s**2).sqrt()
        c_noise = sigma.log() / 4
        return c_skip, c_out, c_in, c_noise

    def denoise(
        self,
        noisy_target: torch.Tensor,  # (B, C, H, W) noisy next frame
        context: torch.Tensor,        # (B, L, C, H, W) past frames
        action: torch.Tensor,         # (B,) action indices
        sigma: torch.Tensor,          # (B,) noise levels
    ) -> torch.Tensor:
        """Single denoising step. Returns predicted clean frame."""
        B = noisy_target.shape[0]
        c_skip, c_out, c_in, c_noise = self._edm_precond(sigma)

        # Build conditioning vector
        sigma_emb = self.fourier(c_noise)                     # (B, fourier_dim)
        act_emb = self.action_emb(action)                     # (B, cond_channels)
        cond = self.cond_proj(torch.cat([sigma_emb, act_emb], dim=-1))  # (B, cond_channels)

        # Concatenate noisy target with context along channel dim
        context_flat = context.reshape(B, -1, *context.shape[3:])  # (B, L*C, H, W)
        x = torch.cat([c_in[:, None, None, None] * noisy_target, context_flat], dim=1)

        # U-Net prediction
        F_x = self.unet(x, cond)

        # EDM output: skip connection + scaled network output
        denoised = c_skip[:, None, None, None] * noisy_target + c_out[:, None, None, None] * F_x
        return denoised

    def training_loss(
        self,
        context: torch.Tensor,   # (B, L, C, H, W)
        target: torch.Tensor,    # (B, C, H, W)
        action: torch.Tensor,    # (B,)
        noise_aug_sigma: float = 0.0,
    ) -> torch.Tensor:
        """Compute EDM training loss."""
        B = target.shape[0]
        device = target.device

        # Sample noise level from log-normal
        sigma = torch.randn(B, device=device).exp() * 0.5  # mean=0, std=0.5 in log space
        sigma = sigma.clamp(0.002, 80.0)

        # Add noise to target
        noise = torch.randn_like(target)
        if self.sigma_offset_noise > 0:
            # Offset noise (shared spatial noise) for better low-frequency learning
            offset = torch.randn(B, target.shape[1], 1, 1, device=device) * self.sigma_offset_noise
            noise = noise + offset
        noisy_target = target + sigma[:, None, None, None] * noise

        # Optionally augment context with noise (GameNGen trick)
        if noise_aug_sigma > 0:
            context = context + torch.randn_like(context) * noise_aug_sigma

        # Predict clean frame
        denoised = self.denoise(noisy_target, context, action, sigma)

        # EDM loss weighting
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        loss = weight[:, None, None, None] * (denoised - target) ** 2
        return loss.mean()

    @torch.no_grad()
    def sample(
        self,
        context: torch.Tensor,   # (B, L, C, H, W)
        action: torch.Tensor,    # (B,)
        num_steps: int = 3,
        sigma_min: float = 0.002,
        sigma_max: float = 5.0,
    ) -> torch.Tensor:
        """Generate next frame using Euler ODE solver."""
        B = context.shape[0]
        device = context.device

        # Sigma schedule (geometric)
        rho = 7.0
        step_indices = torch.arange(num_steps, device=device)
        sigmas = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) *
                  (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        sigmas = torch.cat([sigmas, torch.zeros(1, device=device)])  # append 0

        # Start from noise (derive spatial dims from context)
        H, W = context.shape[3], context.shape[4]
        x = torch.randn(B, self.img_channels, H, W, device=device)
        x = x * sigmas[0]

        for i in range(num_steps):
            sigma = sigmas[i].expand(B)
            denoised = self.denoise(x, context, action, sigma)
            # Euler step
            d = (x - denoised) / sigmas[i]
            x = x + (sigmas[i + 1] - sigmas[i]) * d

        return x.clamp(-1, 1)


def make_denoiser(
    num_actions: int = 10,
    img_size: int = 84,
    num_context_frames: int = 4,
    model_size: str = "small",
) -> Denoiser:
    """Create a denoiser with predefined model sizes."""
    configs = {
        "tiny": dict(channels=[32, 32, 64, 64], depths=[1, 1, 1, 1]),         # ~0.5M
        "small": dict(channels=[64, 64, 64, 64], depths=[2, 2, 2, 2]),        # ~4M (DIAMOND Atari)
        "medium": dict(channels=[64, 128, 128, 128], depths=[2, 2, 2, 2]),    # ~12M
        "large": dict(channels=[128, 128, 256, 256], depths=[2, 2, 2, 2]),    # ~35M
    }
    cfg = configs[model_size]
    return Denoiser(
        num_actions=num_actions,
        img_size=img_size,
        num_context_frames=num_context_frames,
        **cfg,
    )


if __name__ == "__main__":
    model = make_denoiser(num_actions=10, model_size="small")
    params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {params:,}")

    # Test forward pass
    B = 2
    context = torch.randn(B, 4, 3, 84, 84)
    target = torch.randn(B, 3, 84, 84)
    action = torch.randint(0, 10, (B,))

    loss = model.training_loss(context, target, action)
    print(f"Training loss: {loss.item():.4f}")

    # Test sampling
    generated = model.sample(context, action, num_steps=3)
    print(f"Generated shape: {generated.shape}")
