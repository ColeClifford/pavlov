"""Cross-modal retrieval evaluation using cosine similarity."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate_retrieval(
    model,
    dataloader: DataLoader,
    query_mod: str,
    gallery_mod: str,
    k: int = 10,
) -> dict[str, float]:
    """Evaluate cross-modal retrieval with cosine similarity.

    Args:
        model: PavlovModel (or LightningModule with .model attribute).
        dataloader: DataLoader yielding dicts with modality keys and 'label'.
        query_mod: Modality used for queries.
        gallery_mod: Modality used for the gallery.
        k: Top-k for Recall@K computation.

    Returns:
        Dict with 'recall_at_k' and 'mean_average_precision'.
    """
    pavlov_model = getattr(model, "model", model)
    pavlov_model.eval()
    device = next(pavlov_model.parameters()).device

    query_embeddings = []
    gallery_embeddings = []
    labels = []

    for batch in dataloader:
        x_q = batch[query_mod].to(device)
        x_g = batch[gallery_mod].to(device)
        y = batch["label"]

        z_q = pavlov_model.encode(x_q, query_mod)
        z_g = pavlov_model.encode(x_g, gallery_mod)

        query_embeddings.append(z_q.cpu())
        gallery_embeddings.append(z_g.cpu())
        labels.append(y)

    query_embeddings = torch.cat(query_embeddings, dim=0)
    gallery_embeddings = torch.cat(gallery_embeddings, dim=0)
    labels = torch.cat(labels, dim=0)

    # Normalize for cosine similarity
    query_norm = F.normalize(query_embeddings, dim=-1)
    gallery_norm = F.normalize(gallery_embeddings, dim=-1)

    # Similarity matrix: (n_query, n_gallery)
    sim = query_norm @ gallery_norm.T

    # Sort gallery by similarity (descending)
    _, indices = sim.sort(dim=1, descending=True)

    query_labels = labels.unsqueeze(1)  # (N, 1)
    gallery_labels = labels[indices]  # (N, N) reordered

    relevant = (gallery_labels == query_labels).float()

    # Recall@K
    recall_at_k = relevant[:, :k].sum(dim=1).clamp(max=1).mean().item()

    # Mean Average Precision
    cum_relevant = relevant.cumsum(dim=1)
    positions = torch.arange(1, relevant.shape[1] + 1, dtype=torch.float32).unsqueeze(0)
    precision_at_i = cum_relevant / positions
    ap = (precision_at_i * relevant).sum(dim=1) / relevant.sum(dim=1).clamp(min=1)
    mean_ap = ap.mean().item()

    return {
        f"recall_at_{k}": recall_at_k,
        "mean_average_precision": mean_ap,
    }
