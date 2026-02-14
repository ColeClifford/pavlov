"""Embedding visualization with t-SNE."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader


@torch.no_grad()
def plot_embeddings(
    model,
    dataloader: DataLoader,
    save_path: str | Path,
    modalities: list[str] | None = None,
    max_samples: int = 2000,
    perplexity: float = 30.0,
) -> None:
    """Extract embeddings, run t-SNE, and save a scatter plot.

    Points are colored by digit class with different markers per modality.

    Args:
        model: PavlovModel (or LightningModule with .model attribute).
        dataloader: DataLoader yielding dicts with modality keys and 'label'.
        save_path: Path to save the output figure.
        modalities: Modalities to plot. Defaults to all in the first batch.
        max_samples: Max samples per modality to keep plot readable.
        perplexity: t-SNE perplexity parameter.
    """
    pavlov_model = getattr(model, "model", model)
    pavlov_model.eval()
    device = next(pavlov_model.parameters()).device

    embeddings_by_mod: dict[str, list] = {}
    labels_list: list = []

    for batch in dataloader:
        if modalities is None:
            modalities = [k for k in batch if k != "label"]

        y = batch["label"]
        labels_list.append(y.numpy())

        for m in modalities:
            x = batch[m].to(device)
            z = pavlov_model.encode(x, m).cpu().numpy()
            embeddings_by_mod.setdefault(m, []).append(z)

    labels = np.concatenate(labels_list, axis=0)

    # Truncate to max_samples
    n = min(len(labels), max_samples)

    all_z = []
    mod_indices = {}  # modality -> (start, end) indices in combined array
    offset = 0
    for m in modalities:
        z = np.concatenate(embeddings_by_mod[m], axis=0)[:n]
        all_z.append(z)
        mod_indices[m] = (offset, offset + len(z))
        offset += len(z)

    labels = labels[:n]
    all_z = np.concatenate(all_z, axis=0)

    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    coords = tsne.fit_transform(all_z)

    # Plot
    markers = ["o", "^", "s", "D", "v", "<", ">", "p", "*", "h"]
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    for i, m in enumerate(modalities):
        start, end = mod_indices[m]
        marker = markers[i % len(markers)]
        scatter = ax.scatter(
            coords[start:end, 0],
            coords[start:end, 1],
            c=labels,
            cmap="tab10",
            marker=marker,
            alpha=0.6,
            s=15,
            label=m,
            vmin=0,
            vmax=9,
        )

    ax.legend(title="Modality", loc="upper right")
    cbar = fig.colorbar(scatter, ax=ax, ticks=range(10))
    cbar.set_label("Digit Class")
    ax.set_title("Pavlov Embeddings (t-SNE)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved embedding visualization to {save_path}")
