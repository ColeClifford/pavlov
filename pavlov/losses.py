"""Loss functions for Pavlov multimodal training."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def reconstruction_loss(x_recon: Tensor, x_target: Tensor) -> Tensor:
    """Mean squared error reconstruction loss."""
    return F.mse_loss(x_recon, x_target)


def contrastive_loss(
    z_list: list[Tensor],
    labels: Tensor,
    temperature: float = 0.07,
) -> Tensor:
    """InfoNCE-style contrastive loss across modality pairs.

    For each pair of modalities, embeddings with the same label are pulled
    together while embeddings with different labels are pushed apart.

    Args:
        z_list: List of embedding tensors, each (batch, latent_dim).
        labels: Integer class labels of shape (batch,).
        temperature: Softmax temperature scaling.

    Returns:
        Scalar loss averaged over all modality pairs.
    """
    if len(z_list) < 2:
        return torch.tensor(0.0, device=z_list[0].device)

    total_loss = torch.tensor(0.0, device=z_list[0].device)
    n_pairs = 0

    for i in range(len(z_list)):
        for j in range(i + 1, len(z_list)):
            z_i = F.normalize(z_list[i], dim=-1)
            z_j = F.normalize(z_list[j], dim=-1)

            # Cosine similarity matrix: (batch, batch)
            sim = z_i @ z_j.T / temperature

            # Positive mask: same label across modalities
            pos_mask = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B)

            # InfoNCE: for each anchor in z_i, positives are z_j with same label
            # Forward direction: anchor = z_i, candidates = z_j
            loss_ij = _masked_infonce(sim, pos_mask)
            # Backward direction: anchor = z_j, candidates = z_i
            loss_ji = _masked_infonce(sim.T, pos_mask.T)

            total_loss = total_loss + (loss_ij + loss_ji) / 2
            n_pairs += 1

    return total_loss / n_pairs


def _masked_infonce(sim: Tensor, pos_mask: Tensor) -> Tensor:
    """Compute InfoNCE loss given similarity matrix and positive mask."""
    # For each row, compute log-softmax and average over positive entries
    log_prob = F.log_softmax(sim, dim=1)

    # Avoid division by zero if a row has no positives
    pos_count = pos_mask.sum(dim=1).clamp(min=1)
    loss = -(log_prob * pos_mask.float()).sum(dim=1) / pos_count
    return loss.mean()


def orthogonality_loss(rotations: list[Tensor]) -> Tensor:
    """Encourage rotation matrices to be orthogonal to each other.

    loss = sum_{i<j} |trace(R_i^T @ R_j)|^2 / n_pairs

    Args:
        rotations: List of rotation matrices, each (latent_dim, latent_dim).

    Returns:
        Scalar orthogonality loss.
    """
    if len(rotations) < 2:
        return torch.tensor(0.0, device=rotations[0].device)

    total_loss = torch.tensor(0.0, device=rotations[0].device)
    n_pairs = 0

    for i in range(len(rotations)):
        for j in range(i + 1, len(rotations)):
            trace_val = torch.trace(rotations[i].T @ rotations[j])
            total_loss = total_loss + trace_val**2
            n_pairs += 1

    return total_loss / n_pairs
