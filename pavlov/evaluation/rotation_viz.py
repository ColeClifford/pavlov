"""Rotation matrix visualization for TensorBoard."""

from __future__ import annotations

import torch


def log_rotation_matrices(model, writer, step: int, modalities: list[str] | None = None) -> None:
    """Log learned rotation matrices R_m and their relationships as heatmap images.

    Logs:
    - geometry/R_{modality} for each modality
    - geometry/R_relative_{a}_{b} for R_a^T @ R_b (relative rotation)
    - geometry/orthogonality_check_{modality}: R @ R.T (should be identity)
    """
    pavlov_model = getattr(model, "model", model)
    if modalities is None:
        modalities = list(pavlov_model.modalities)

    if not hasattr(writer, "add_image"):
        return

    for m in modalities:
        R = pavlov_model.get_rotation(m).detach().cpu()
        # Normalize for visibility: scale to [0, 1] for display
        r_min, r_max = R.min().item(), R.max().item()
        if r_max - r_min > 1e-8:
            R_norm = (R - r_min) / (r_max - r_min)
        else:
            R_norm = torch.zeros_like(R)
        img = R_norm.unsqueeze(0).unsqueeze(0)  # (1, 1, d, d)
        writer.add_image(f"geometry/R_{m}", img, step)

        # Orthogonality check: R @ R.T should be I (identity)
        ortho = R @ R.T
        ortho_norm = (ortho + 1) / 2  # Map [-1,1] to [0,1] for visibility
        writer.add_image(
            f"geometry/orthogonality_check_{m}",
            ortho_norm.unsqueeze(0).unsqueeze(0),
            step,
        )

    # Relative rotations between modality pairs
    for i, m1 in enumerate(modalities):
        for m2 in modalities[i + 1 :]:
            R1 = pavlov_model.get_rotation(m1).detach().cpu()
            R2 = pavlov_model.get_rotation(m2).detach().cpu()
            R_rel = (R1.T @ R2).unsqueeze(0).unsqueeze(0)
            r_min, r_max = R_rel.min().item(), R_rel.max().item()
            if r_max - r_min > 1e-8:
                R_rel_norm = (R_rel - r_min) / (r_max - r_min)
            else:
                R_rel_norm = torch.zeros_like(R_rel)
            writer.add_image(f"geometry/R_relative_{m1}_{m2}", R_rel_norm, step)
