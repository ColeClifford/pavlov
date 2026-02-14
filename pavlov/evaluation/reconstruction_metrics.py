"""Reconstruction quality metrics for evaluation."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


@torch.no_grad()
def compute_reconstruction_mse(
    model,
    dataloader: DataLoader,
    modalities: list[str],
) -> dict[str, float]:
    """Compute per-modality same-modal and cross-modal MSE over a dataloader.

    Returns:
        Dict with keys like 'vision/same_modal', 'vision/cross_from_audio',
        'audio/same_modal', 'audio/cross_from_vision'.
    """
    pavlov_model = getattr(model, "model", model)
    pavlov_model.eval()
    device = next(pavlov_model.parameters()).device

    mse_sums: dict[str, float] = {}
    mse_counts: dict[str, int] = {}

    for batch in dataloader:
        z = {}
        for m in modalities:
            if m in batch:
                x = batch[m].to(device)
                z[m] = pavlov_model.encode(x, m)

        for tgt in modalities:
            if tgt not in batch:
                continue
            # Same-modal
            key = f"{tgt}/same_modal"
            x_recon = pavlov_model.decode(z[tgt], tgt)
            mse = F.mse_loss(x_recon, batch[tgt].to(device)).item()
            mse_sums[key] = mse_sums.get(key, 0) + mse * batch[tgt].shape[0]
            mse_counts[key] = mse_counts.get(key, 0) + batch[tgt].shape[0]

            # Cross-modal
            for src in modalities:
                if src == tgt:
                    continue
                key = f"{tgt}/cross_from_{src}"
                x_recon = pavlov_model.decode(z[src], tgt)
                mse = F.mse_loss(x_recon, batch[tgt].to(device)).item()
                mse_sums[key] = mse_sums.get(key, 0) + mse * batch[tgt].shape[0]
                mse_counts[key] = mse_counts.get(key, 0) + batch[tgt].shape[0]

    return {k: mse_sums[k] / mse_counts[k] for k in mse_sums}
