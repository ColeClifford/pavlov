"""Cross-modal transfer evaluation: train on one modality, test on another."""

from __future__ import annotations

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate_cross_modal_transfer(
    model,
    dataloader: DataLoader,
    train_modality: str,
    test_modality: str,
) -> dict[str, float]:
    """Train a linear classifier on embeddings from one modality, evaluate on another.

    Args:
        model: PavlovModel (or LightningModule with .model attribute).
        dataloader: DataLoader yielding dicts with modality keys and 'label'.
        train_modality: Modality to use for training the classifier.
        test_modality: Modality to use for testing the classifier.

    Returns:
        Dict with 'train_accuracy' and 'transfer_accuracy'.
    """
    pavlov_model = getattr(model, "model", model)
    pavlov_model.eval()
    device = next(pavlov_model.parameters()).device

    train_embeddings = []
    test_embeddings = []
    labels = []

    for batch in dataloader:
        x_train = batch[train_modality].to(device)
        x_test = batch[test_modality].to(device)
        y = batch["label"]

        z_train = pavlov_model.encode(x_train, train_modality)
        z_test = pavlov_model.encode(x_test, test_modality)

        train_embeddings.append(z_train.cpu().numpy())
        test_embeddings.append(z_test.cpu().numpy())
        labels.append(y.numpy())

    train_embeddings = np.concatenate(train_embeddings, axis=0)
    test_embeddings = np.concatenate(test_embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)

    clf = LogisticRegression(max_iter=1000, solver="lbfgs")
    clf.fit(train_embeddings, labels)

    train_accuracy = clf.score(train_embeddings, labels)
    transfer_accuracy = clf.score(test_embeddings, labels)

    return {
        "train_accuracy": train_accuracy,
        "transfer_accuracy": transfer_accuracy,
    }
