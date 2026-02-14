"""Modality gap and alignment metrics."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


@torch.no_grad()
def compute_modality_alignment(
    model,
    dataloader: DataLoader,
    modalities: list[str],
    max_samples: int = 2000,
) -> dict[str, float]:
    """Compute modality alignment metrics: cross-modal centroid similarity and distance.

    For each digit class, computes mean embedding per modality. Measures how well
    vision and audio class centroids align (cosine similarity, distance).

    Returns:
        Dict with 'mean_cross_modal_similarity', 'vision_audio_centroid_distance',
        and per-class metrics if useful.
    """
    pavlov_model = getattr(model, "model", model)
    pavlov_model.eval()
    device = next(pavlov_model.parameters()).device

    embeddings_by_mod: dict[str, list] = {m: [] for m in modalities}
    labels_list: list[torch.Tensor] = []

    n = 0
    for batch in dataloader:
        if n >= max_samples:
            break
        y = batch["label"]
        labels_list.append(y)
        for m in modalities:
            if m in batch:
                x = batch[m].to(device)
                z = pavlov_model.encode(x, m).cpu()
                embeddings_by_mod[m].append(z)
        n += batch["label"].shape[0]

    if n == 0:
        return {}

    labels = torch.cat(labels_list, dim=0)[:max_samples]
    z_dict = {
        m: torch.cat(embeddings_by_mod[m], dim=0)[:max_samples]
        for m in modalities
        if embeddings_by_mod[m]
    }

    if len(z_dict) < 2:
        return {}

    # Normalize for cosine similarity
    z_norm = {m: F.normalize(z_dict[m], dim=-1) for m in z_dict}

    # Per-class centroids
    n_classes = int(labels.max().item()) + 1
    centroid_sims = []
    centroid_dists = []

    for c in range(n_classes):
        mask = labels == c
        if mask.sum() < 2:
            continue
        mod_keys = list(z_norm.keys())
        centroids = {}
        for m in mod_keys:
            centroids[m] = z_norm[m][mask].mean(dim=0)
            centroids[m] = F.normalize(centroids[m].unsqueeze(0), dim=-1).squeeze(0)

        for i, m1 in enumerate(mod_keys):
            for m2 in mod_keys[i + 1 :]:
                sim = (centroids[m1] * centroids[m2]).sum().item()
                dist = 1 - sim  # Cosine distance
                centroid_sims.append(sim)
                centroid_dists.append(dist)

    if not centroid_sims:
        return {}

    return {
        "mean_cross_modal_similarity": sum(centroid_sims) / len(centroid_sims),
        "vision_audio_centroid_distance": sum(centroid_dists) / len(centroid_dists)
        if centroid_dists
        else 0.0,
    }


def log_modality_alignment(
    model,
    dataloader: DataLoader,
    writer,
    step: int,
    modalities: list[str] | None = None,
) -> None:
    """Compute modality alignment and log to TensorBoard."""
    if modalities is None:
        modalities = list(getattr(model, "model", model).modalities)
    if not hasattr(writer, "add_scalar"):
        return
    metrics = compute_modality_alignment(model, dataloader, modalities)
    for key, val in metrics.items():
        writer.add_scalar(f"modality_gap/{key}", val, step)
