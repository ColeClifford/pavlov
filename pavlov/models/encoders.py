from abc import ABC, abstractmethod

import torch.nn as nn


class ModalityEncoder(ABC, nn.Module):
    """Abstract base class for modality-specific encoders."""

    @abstractmethod
    def forward(self, x):
        """Encode raw input into token embeddings.

        Args:
            x: Raw modality input tensor.

        Returns:
            Tensor of shape (batch, embed_dim).
        """

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Dimensionality of the encoder output."""


class VisionEncoder(ModalityEncoder):
    """CNN encoder for grayscale or RGB images of arbitrary spatial size.

    For small inputs (< 64px), uses the original AV-MNIST architecture
    (2 conv layers, direct flatten) for checkpoint compatibility.
    For larger inputs, uses a deeper stack with BatchNorm and adaptive
    pooling so the FC layer size is independent of spatial resolution.
    """

    def __init__(
        self,
        embed_dim: int,
        input_channels: int = 1,
        input_size: int = 28,
    ):
        super().__init__()
        self._embed_dim = embed_dim
        self._use_pool = input_size >= 64

        if self._use_pool:
            self.conv = nn.Sequential(
                nn.Conv2d(input_channels, 32, 3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
            )
            self.pool = nn.AdaptiveAvgPool2d((4, 4))
            self.fc = nn.Linear(128 * 4 * 4, embed_dim)
        else:
            # Original AV-MNIST architecture — preserves checkpoint compat
            self.conv = nn.Sequential(
                nn.Conv2d(input_channels, 32, 3, stride=2, padding=1),  # -> 32x14x14
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),  # -> 64x7x7
                nn.ReLU(),
            )
            self.pool = None
            self.fc = nn.Linear(64 * 7 * 7, embed_dim)

    @property
    def output_dim(self) -> int:
        return self._embed_dim

    def forward(self, x):
        h = self.conv(x)
        if self.pool is not None:
            h = self.pool(h)
        h = h.flatten(1)
        return self.fc(h)


class AudioEncoder(ModalityEncoder):
    """CNN encoder for 1x112x112 spectrograms.

    Deeper than the vision encoder to handle the 16x-larger input.
    Uses BatchNorm for spectrogram normalization, adaptive pooling to
    decouple spatial resolution from the FC layer, and dropout for
    regularization.
    """

    def __init__(self, embed_dim: int, dropout: float = 0.3):
        super().__init__()
        self._embed_dim = embed_dim
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),    # -> 32x56x56
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),   # -> 64x28x28
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # -> 128x14x14
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), # -> 256x7x7
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        # Adaptive pooling decouples feature map size from FC input
        self.pool = nn.AdaptiveAvgPool2d((4, 4))  # -> 256x4x4 = 4096
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256 * 4 * 4, embed_dim),
        )

    @property
    def output_dim(self) -> int:
        return self._embed_dim

    def forward(self, x):
        h = self.conv(x)
        h = self.pool(h)
        h = h.flatten(1)
        return self.fc(h)


def build_encoder(modality: str, cfg) -> ModalityEncoder:
    """Factory function to create a modality encoder.

    Args:
        modality: One of "vision" or "audio".
        cfg: Config object with an embed_dim attribute and optional
            vision_encoder / audio_encoder sub-configs.

    Returns:
        A ModalityEncoder instance.
    """
    embed_dim = cfg.embed_dim
    if modality == "vision":
        ve_cfg = getattr(cfg, "vision_encoder", None)
        input_channels = getattr(ve_cfg, "input_channels", 1) if ve_cfg else 1
        input_size = getattr(ve_cfg, "input_size", 28) if ve_cfg else 28
        return VisionEncoder(
            embed_dim,
            input_channels=input_channels,
            input_size=input_size,
        )
    elif modality == "audio":
        return AudioEncoder(embed_dim)
    else:
        raise ValueError(f"Unknown modality: {modality}")
