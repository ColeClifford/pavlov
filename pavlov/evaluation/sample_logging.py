"""Sample reconstruction logging for TensorBoard."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from pavlov.evaluation.audio_utils import AVMNIST_SR, spectrogram_to_audio


@torch.no_grad()
def log_sample_reconstructions(
    model,
    batch: dict[str, torch.Tensor],
    writer,
    step: int,
    n_samples: int = 8,
    modalities: list[str] | None = None,
    log_audio: bool = True,
    log_comparison_grids: bool = True,
) -> None:
    """Log ground truth, same-modal, and cross-modal reconstructions to TensorBoard.

    Args:
        model: PavlovModel or LightningModule with .model attribute.
        batch: Batch dict with modality keys (e.g. 'vision', 'audio') and 'label'.
        writer: TensorBoard SummaryWriter or Lightning logger.experiment (has add_image).
        step: Global step / epoch for logging.
        n_samples: Number of samples to include in each grid.
        modalities: Modalities to log. Defaults to all in batch except 'label'.
        log_audio: If True and writer has add_audio, log audio clips for audio modality.
        log_comparison_grids: If True, log side-by-side comparison grids (GT | same | cross).
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

    # Side-by-side comparison grids: [GT | same_modal | cross_modal] per sample
    if log_comparison_grids and hasattr(writer, "add_image"):
        n_comp = min(4, n)
        for tgt_mod in modalities:
            if tgt_mod not in batch:
                continue
            src_mod = next((m for m in modalities if m != tgt_mod), None)
            if src_mod is None or src_mod not in z:
                continue
            rows = []
            for i in range(n_comp):
                gt_i = batch[tgt_mod][i : i + 1].cpu()
                same_i = pavlov_model.decode(z[tgt_mod][i : i + 1], tgt_mod).cpu()
                cross_i = pavlov_model.decode(z[src_mod][i : i + 1], tgt_mod).cpu()
                row = torch.cat([gt_i, same_i, cross_i], dim=0)  # (3, C, H, W)
                rows.append(row)
            comp_grid = torch.cat(rows, dim=0)  # (n_comp*3, C, H, W)
            grid = make_grid(comp_grid, nrow=3, normalize=True, scale_each=True)
            _log_image(writer, f"samples/{tgt_mod}/comparison_gt_same_cross", grid, step)

    # Audio playback for audio modality (spectrograms)
    if log_audio and "audio" in batch and hasattr(writer, "add_audio"):
        n_audio = min(4, n)  # Log up to 4 audio clips
        for i in range(n_audio):
            # Ground truth
            spec_gt = batch["audio"][i].cpu().numpy()
            audio_gt = spectrogram_to_audio(spec_gt)
            _log_audio(writer, f"audio/audio/ground_truth/{i}", audio_gt, step)
            # Same-modal recon
            spec_same = pavlov_model.decode(z["audio"][i : i + 1], "audio").cpu().numpy().squeeze(0)
            audio_same = spectrogram_to_audio(spec_same)
            _log_audio(writer, f"audio/audio/same_modal_recon/{i}", audio_same, step)
            # Cross-modal from each other modality
            for src_mod in modalities:
                if src_mod == "audio":
                    continue
                if src_mod not in z:
                    continue
                spec_cross = pavlov_model.decode(z[src_mod][i : i + 1], "audio").cpu().numpy().squeeze(0)
                audio_cross = spectrogram_to_audio(spec_cross)
                _log_audio(writer, f"audio/audio/cross_modal_from_{src_mod}/{i}", audio_cross, step)


def _log_image(writer, tag: str, img_tensor: torch.Tensor, step: int) -> None:
    """Log image to writer. Handles SummaryWriter and Lightning logger.experiment."""
    # Both TensorBoard SummaryWriter and Lightning's TensorBoardLogger.experiment
    # use add_image(tag, img_tensor, global_step). img_tensor: CHW or NCHW.
    if hasattr(writer, "add_image"):
        writer.add_image(tag, img_tensor, step)
    else:
        raise TypeError(f"Writer {type(writer)} has no add_image method")


def _log_audio(
    writer,
    tag: str,
    audio: np.ndarray,
    step: int,
    sample_rate: int = AVMNIST_SR,
) -> None:
    """Log audio to TensorBoard. Expects audio in [-1, 1], shape (L,) or (C, L)."""
    audio_tensor = torch.from_numpy(audio).float()
    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)  # (1, L)
    writer.add_audio(tag, audio_tensor, step, sample_rate=sample_rate)


def log_samples_from_dataloader(
    model,
    dataloader: DataLoader,
    writer,
    step: int = 0,
    n_samples: int = 8,
    modalities: list[str] | None = None,
    log_audio: bool = True,
    log_comparison_grids: bool = True,
) -> None:
    """Fetch one batch from dataloader and log sample reconstructions."""
    batch = next(iter(dataloader))
    log_sample_reconstructions(
        model,
        batch,
        writer,
        step,
        n_samples,
        modalities,
        log_audio=log_audio,
        log_comparison_grids=log_comparison_grids,
    )
