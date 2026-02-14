from __future__ import annotations

import torch
import torch.nn as nn

from pavlov.models.encoders import build_encoder
from pavlov.models.decoders import build_decoder
from pavlov.utils.rotation import skew_symmetric_to_rotation


class PavlovModel(nn.Module):
    """Core Pavlov model: shared embeddings with geometric modality conditioning.

    Pipeline:
        Encode: raw_input -> Encoder_m(x) -> tokens (embed_dim) -> W(tokens) -> z_shared (latent_dim) -> z_shared @ R_m^T -> z_m
        Decode: z -> W_dec(z) -> h (embed_dim) -> Decoder_m(h) -> reconstruction
    """

    def __init__(
        self,
        modalities: list[str],
        embed_dim: int,
        latent_dim: int,
        model_cfg,
        init_rotation_std: float = 0.01,
    ):
        super().__init__()
        self.modalities = modalities
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim

        # Shared projection layers (no bias)
        self.W = nn.Linear(embed_dim, latent_dim, bias=False)
        self.W_dec = nn.Linear(latent_dim, embed_dim, bias=False)

        # Per-modality encoders and decoders
        self.encoders = nn.ModuleDict({m: build_encoder(m, model_cfg) for m in modalities})
        self.decoders = nn.ModuleDict({m: build_decoder(m, model_cfg) for m in modalities})

        # Per-modality rotation parameters (Lie algebra so(latent_dim))
        n_params = latent_dim * (latent_dim - 1) // 2
        self.rotation_params = nn.ParameterDict(
            {m: nn.Parameter(torch.randn(n_params) * init_rotation_std) for m in modalities}
        )

    def get_rotation(self, modality: str) -> torch.Tensor:
        """Get the rotation matrix R_m for a given modality.

        Returns:
            Rotation matrix of shape (latent_dim, latent_dim) in SO(latent_dim).
        """
        return skew_symmetric_to_rotation(self.rotation_params[modality], self.latent_dim)

    def compute_rotations(self, modalities: list[str] | None = None) -> dict[str, torch.Tensor]:
        """Pre-compute rotation matrices for all (or given) modalities.

        Call once per step and pass the result to encode/decode to avoid
        redundant matrix_exp computations.

        Returns:
            Dict mapping modality name to its rotation matrix.
        """
        if modalities is None:
            modalities = self.modalities
        return {m: self.get_rotation(m) for m in modalities}

    def encode(
        self,
        x: torch.Tensor,
        modality: str,
        rotation: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode raw input to modality-conditioned latent representation.

        Args:
            x: Raw input tensor for the given modality.
            modality: The modality name ("vision" or "audio").
            rotation: Optional pre-computed rotation matrix. If None, computed
                on the fly (less efficient when called multiple times per step).

        Returns:
            z_m of shape (batch, latent_dim).
        """
        tokens = self.encoders[modality](x)  # (batch, embed_dim)
        z_shared = self.W(tokens)  # (batch, latent_dim)
        R = rotation if rotation is not None else self.get_rotation(modality)
        z_m = z_shared @ R.T  # (batch, latent_dim)
        return z_m

    def decode(self, z: torch.Tensor, modality: str) -> torch.Tensor:
        """Decode latent representation to reconstructed input.

        Args:
            z: Latent tensor of shape (batch, latent_dim).
            modality: The target modality for reconstruction.

        Returns:
            Reconstructed input tensor.
        """
        h = self.W_dec(z)  # (batch, embed_dim)
        return self.decoders[modality](h)

    def forward(
        self, inputs: dict[str, torch.Tensor]
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Full forward pass: encode each modality, decode to all modalities.

        Args:
            inputs: Dict mapping modality name to input tensor.

        Returns:
            Nested dict: result[source_modality][target_modality] = reconstructed tensor.
            Also includes "z" key: result["z"][modality] = latent embedding.
        """
        # Pre-compute rotations once for the entire forward pass
        rotations = self.compute_rotations(list(inputs.keys()))

        latents = {}
        for modality, x in inputs.items():
            latents[modality] = self.encode(x, modality, rotation=rotations[modality])

        reconstructions = {}
        for src_mod, z in latents.items():
            reconstructions[src_mod] = {}
            for tgt_mod in self.modalities:
                reconstructions[src_mod][tgt_mod] = self.decode(z, tgt_mod)

        return {"reconstructions": reconstructions, "z": latents}
