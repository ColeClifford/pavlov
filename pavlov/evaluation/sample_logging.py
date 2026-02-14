"""Sample reconstruction logging for TensorBoard."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid


@torch.no_grad()
def log_sample_reconstructions(
    model,
    batch: dict[str, torch.Tensor],
    writer,
    step: int,
    n_samples: int = 8,
    modalities: list[str] | None = None,
) -> None:
    """Log ground truth, same-modal, and cross-modal reconstructions to TensorBoard.

    Args:
        model: PavlovModel or LightningModule with .model attribute.
        batch: Batch dict with modality keys (e.g. 'vision', 'audio') and 'label'.
        writer: TensorBoard SummaryWriter or Lightning logger.experiment (has add_image).
        step: Global step / epoch for logging.
        n_samples: Number of samples to include in each grid.
        modalities: Modalities to log. Defaults to all in batch except 'label'.
    """
    pavlov_model = getattr(model, "model", model)
    pavlov_model.eval()
    device = next(pavlov_model.parameters()).device

    if modalities is None:
        modalities = [k for k in batch if k != "label"]

    first_mod = next((m for m in modalities if m in batch), None)
    if first_mod is None:
        return
    n = min(n_samples, batch[first_mod].shape[0])

    # Encode all modalities
    z = {}
    for m in modalities:
        if m in batch:
            x = batch[m].to(device)
            z[m] = pavlov_model.encode(x, m)

    # Build and log grids for each modality
    for tgt_mod in modalities:
        if tgt_mod not in batch:
            continue

        # Ground truth
        gt = batch[tgt_mod][:n].cpu()
        grid_gt = make_grid(gt, nrow=min(n, 8), normalize=True, scale_each=True)
        _log_image(writer, f"samples/{tgt_mod}/ground_truth", grid_gt, step)

        # Same-modal reconstruction
        recon_same = pavlov_model.decode(z[tgt_mod][:n], tgt_mod).cpu()
        grid_same = make_grid(recon_same, nrow=min(n, 8), normalize=True, scale_each=True)
        _log_image(writer, f"samples/{tgt_mod}/same_modal_recon", grid_same, step)

        # Cross-modal reconstructions (from each other modality)
        for src_mod in modalities:
            if src_mod == tgt_mod:
                continue
            if src_mod not in z:
                continue
            recon_cross = pavlov_model.decode(z[src_mod][:n], tgt_mod).cpu()
            grid_cross = make_grid(recon_cross, nrow=min(n, 8), normalize=True, scale_each=True)
            _log_image(writer, f"samples/{tgt_mod}/cross_modal_from_{src_mod}", grid_cross, step)


def _log_image(writer, tag: str, img_tensor: torch.Tensor, step: int) -> None:
    """Log image to writer. Handles SummaryWriter and Lightning logger.experiment."""
    # Both TensorBoard SummaryWriter and Lightning's TensorBoardLogger.experiment
    # use add_image(tag, img_tensor, global_step). img_tensor: CHW or NCHW.
    if hasattr(writer, "add_image"):
        writer.add_image(tag, img_tensor, step)
    else:
        raise TypeError(f"Writer {type(writer)} has no add_image method")


def log_samples_from_dataloader(
    model,
    dataloader: DataLoader,
    writer,
    step: int = 0,
    n_samples: int = 8,
    modalities: list[str] | None = None,
) -> None:
    """Fetch one batch from dataloader and log sample reconstructions."""
    batch = next(iter(dataloader))
    log_sample_reconstructions(model, batch, writer, step, n_samples, modalities)
