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
    """Transposed-CNN decoder producing 1x28x28 images."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.fc = nn.Linear(embed_dim, 64 * 7 * 7)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # -> 32x14x14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),  # -> 1x28x28
            nn.Sigmoid(),
        )

    def forward(self, h):
        h = self.fc(h)
        h = h.view(-1, 64, 7, 7)
        return self.deconv(h)


class AudioDecoder(ModalityDecoder):
    """Transposed-CNN decoder producing 1x112x112 spectrograms."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.fc = nn.Linear(embed_dim, 128 * 14 * 14)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # -> 64x28x28
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # -> 32x56x56
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),  # -> 1x112x112
            nn.Sigmoid(),
        )

    def forward(self, h):
        h = self.fc(h)
        h = h.view(-1, 128, 14, 14)
        return self.deconv(h)


def build_decoder(modality: str, cfg) -> ModalityDecoder:
    """Factory function to create a modality decoder.

    Args:
        modality: One of "vision" or "audio".
        cfg: Config object with an embed_dim attribute.

    Returns:
        A ModalityDecoder instance.
    """
    embed_dim = cfg.embed_dim
    if modality == "vision":
        return VisionDecoder(embed_dim)
    elif modality == "audio":
        return AudioDecoder(embed_dim)
    else:
        raise ValueError(f"Unknown modality: {modality}")
