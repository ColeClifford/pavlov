"""Missing-modality robustness evaluation."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate_missing_modality(
    model,
    dataloader: DataLoader,
    available_modality: str,
    modalities: list[str],
) -> dict[str, float]:
    """Evaluate when only one modality is available.

    Trains a linear classifier on embeddings from the available modality and
    tests on the same. Also measures reconstruction quality when "filling in"
    the missing modality by encoding from available and decoding to missing.

    Args:
        model: PavlovModel or LightningModule.
        dataloader: DataLoader with modality keys and 'label'.
        available_modality: The only modality available (e.g. 'vision' or 'audio').
        modalities: Full list of modalities (e.g. ['vision', 'audio']).

    Returns:
        Dict with 'accuracy' (linear probe on available modality) and
        'cross_recon_mse' (MSE when decoding to missing modality from available).
    """
    pavlov_model = getattr(model, "model", model)
    pavlov_model.eval()
    device = next(pavlov_model.parameters()).device

    embeddings = []
    labels_list = []
    batch_for_recon = None

    for batch in dataloader:
        x = batch[available_modality].to(device)
        y = batch["label"]
        z = pavlov_model.encode(x, available_modality)
        embeddings.append(z.cpu().numpy())
        labels_list.append(y.numpy())
        if batch_for_recon is None:
            batch_for_recon = {k: v[:8] for k, v in batch.items()}

    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    # Linear probe on available modality
    clf = LogisticRegression(max_iter=1000, solver="lbfgs")
    clf.fit(embeddings, labels)
    accuracy = clf.score(embeddings, labels)

    # Cross-modal reconstruction: encode from available, decode to missing
    missing_modalities = [m for m in modalities if m != available_modality]
    cross_recon_mses = []
    for missing_mod in missing_modalities:
        if missing_mod not in batch_for_recon:
            continue
        x_avail = batch_for_recon[available_modality].to(device)
        x_missing_gt = batch_for_recon[missing_mod].to(device)
        z = pavlov_model.encode(x_avail, available_modality)
        x_missing_recon = pavlov_model.decode(z, missing_mod)
        mse = torch.nn.functional.mse_loss(x_missing_recon, x_missing_gt).item()
        cross_recon_mses.append(mse)

    return {
        "accuracy": accuracy,
        "cross_recon_mse": np.mean(cross_recon_mses) if cross_recon_mses else 0.0,
    }
