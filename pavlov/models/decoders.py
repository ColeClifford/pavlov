from abc import ABC, abstractmethod

import torch.nn as nn


class ModalityDecoder(ABC, nn.Module):
    """Abstract base class for modality-specific decoders."""

    @abstractmethod
    def forward(self, h):
        """Decode embedding back to reconstructed input.

        Args:
            h: Embedding tensor of shape (batch, embed_dim).

        Returns:
            Reconstructed input tensor.
        """


class VisionDecoder(ModalityDecoder):
    """Transposed-CNN decoder producing images of configurable size.

    For small outputs (< 64px), uses the original AV-MNIST architecture
    for checkpoint compatibility. For larger outputs, uses a deeper stack
    with BatchNorm and adaptive resize to hit the exact target size.
    """

    def __init__(
        self,
        embed_dim: int,
        output_channels: int = 1,
        output_size: int = 28,
    ):
        super().__init__()
        self._use_resize = output_size >= 64

        if self._use_resize:
            self.fc = nn.Linear(embed_dim, 128 * 4 * 4)
            self._reshape = (-1, 128, 4, 4)
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # -> 64x8x8
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # -> 32x16x16
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.ConvTranspose2d(32, output_channels, 4, stride=2, padding=1),  # -> CxHxW
                nn.Sigmoid(),
            )
            self.resize = nn.AdaptiveAvgPool2d((output_size, output_size))
        else:
            # Original AV-MNIST architecture — preserves checkpoint compat
            self.fc = nn.Linear(embed_dim, 64 * 7 * 7)
            self._reshape = (-1, 64, 7, 7)
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # -> 32x14x14
                nn.ReLU(),
                nn.ConvTranspose2d(32, output_channels, 4, stride=2, padding=1),  # -> 1x28x28
                nn.Sigmoid(),
            )
            self.resize = None

    def forward(self, h):
        h = self.fc(h)
        h = h.view(*self._reshape)
        h = self.deconv(h)
        if self.resize is not None:
            h = self.resize(h)
        return h


class AudioDecoder(ModalityDecoder):
    """Transposed-CNN decoder producing 1x112x112 spectrograms.

    Mirrors the upgraded AudioEncoder: deeper, with BatchNorm, starting
    from a 256x4x4 spatial layout to match the encoder's adaptive pool.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.fc = nn.Linear(embed_dim, 256 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # -> 128x8x8
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # -> 64x16x16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),    # -> 32x32x32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),    # -> 16x64x64
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),     # -> 1x128x128
            nn.Sigmoid(),
        )
        # Crop/interpolate to exact 112x112 target size
        self.resize = nn.AdaptiveAvgPool2d((112, 112))

    def forward(self, h):
        h = self.fc(h)
        h = h.view(-1, 256, 4, 4)
        h = self.deconv(h)
        return self.resize(h)


def build_decoder(modality: str, cfg) -> ModalityDecoder:
    """Factory function to create a modality decoder.

    Args:
        modality: One of "vision" or "audio".
        cfg: Config object with an embed_dim attribute and optional
            vision_decoder / audio_decoder sub-configs.

    Returns:
        A ModalityDecoder instance.
    """
    embed_dim = cfg.embed_dim
    if modality == "vision":
        vd_cfg = getattr(cfg, "vision_decoder", None)
        output_channels = getattr(vd_cfg, "output_channels", 1) if vd_cfg else 1
        output_size = getattr(vd_cfg, "output_size", 28) if vd_cfg else 28
        return VisionDecoder(
            embed_dim,
            output_channels=output_channels,
            output_size=output_size,
        )
    elif modality == "audio":
        return AudioDecoder(embed_dim)
    else:
        raise ValueError(f"Unknown modality: {modality}")
