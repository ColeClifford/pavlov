"""PyTorch Lightning module for Pavlov training."""

from __future__ import annotations

import math

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from pavlov.evaluation.modality_alignment import log_modality_alignment
from pavlov.evaluation.visualization import log_tsne_embeddings
from pavlov.evaluation.rotation_viz import log_rotation_matrices
from pavlov.evaluation.sample_logging import log_sample_reconstructions
from pavlov.losses import contrastive_loss, orthogonality_loss, reconstruction_loss
from pavlov.models.pavlov_model import PavlovModel


class PavlovLightningModule(pl.LightningModule):
    """Lightning wrapper for the Pavlov multimodal model."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg

        modalities = list(cfg.modalities)
        self.modalities = modalities

        self.model = PavlovModel(
            modalities=modalities,
            embed_dim=cfg.model.embed_dim,
            latent_dim=cfg.model.latent_dim,
            model_cfg=cfg.model,
            init_rotation_std=cfg.model.init_rotation_std,
        )

        lw = cfg.training.loss_weights
        self.w_same = lw.same_modal_recon
        self.w_cross = lw.cross_modal_recon
        self.w_contrastive = lw.contrastive
        self.w_ortho = lw.orthogonality

    def _shared_step(self, batch: dict, prefix: str) -> torch.Tensor:
        # Encode all modalities
        z = {}
        for m in self.modalities:
            if m in batch:
                z[m] = self.model.encode(batch[m], m)

        total_loss = torch.tensor(0.0, device=self.device)

        # Same-modal reconstruction
        same_loss = torch.tensor(0.0, device=self.device)
        modality_keys = list(z.keys())
        for m in modality_keys:
            x_recon = self.model.decode(z[m], m)
            mse = reconstruction_loss(x_recon, batch[m])
            same_loss = same_loss + mse
            self.log(f"{prefix}/mse/{m}/same_modal", mse, prog_bar=False)
        same_loss = same_loss / max(len(z), 1)
        total_loss = total_loss + self.w_same * same_loss
        self.log(f"{prefix}/same_modal_recon", same_loss, prog_bar=False)

        # Cross-modal reconstruction
        cross_loss = torch.tensor(0.0, device=self.device)
        n_cross = 0
        for src in modality_keys:
            for tgt in modality_keys:
                if src != tgt:
                    x_recon = self.model.decode(z[src], tgt)
                    mse = reconstruction_loss(x_recon, batch[tgt])
                    cross_loss = cross_loss + mse
                    self.log(f"{prefix}/mse/{tgt}/cross_from_{src}", mse, prog_bar=False)
                    n_cross += 1
        if n_cross > 0:
            cross_loss = cross_loss / n_cross
        total_loss = total_loss + self.w_cross * cross_loss
        self.log(f"{prefix}/cross_modal_recon", cross_loss, prog_bar=False)

        # Optional contrastive loss
        if self.w_contrastive > 0 and "label" in batch and len(z) >= 2:
            z_list = [z[m] for m in modality_keys]
            c_loss = contrastive_loss(z_list, batch["label"])
            total_loss = total_loss + self.w_contrastive * c_loss
            self.log(f"{prefix}/contrastive", c_loss, prog_bar=False)

        # Optional orthogonality loss
        if self.w_ortho > 0:
            rotations = [self.model.get_rotation(m) for m in modality_keys]
            o_loss = orthogonality_loss(rotations)
            total_loss = total_loss + self.w_ortho * o_loss
            self.log(f"{prefix}/orthogonality", o_loss, prog_bar=False)

        self.log(f"{prefix}/total_loss", total_loss, prog_bar=True)
        return total_loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        if batch_idx == 0:
            self._val_sample_batch = {k: v.detach() for k, v in batch.items()}
        return self._shared_step(batch, "val")

    def on_validation_epoch_end(self) -> None:
        self._log_sample_reconstructions()

    @rank_zero_only
    def _log_sample_reconstructions(self) -> None:
        """Log sample reconstructions to TensorBoard if supported."""
        log_every = getattr(
            self.cfg.training, "log_samples_every_n_epochs", 1
        )
        if (self.current_epoch + 1) % log_every != 0:
            return
        if not hasattr(self, "_val_sample_batch"):
            return
        if self.logger is None:
            return
        experiment = getattr(self.logger, "experiment", None)
        if experiment is None or not hasattr(experiment, "add_image"):
            return
        log_audio = getattr(self.cfg.training, "log_audio_samples", True)
        log_sample_reconstructions(
            self,
            self._val_sample_batch,
            experiment,
            step=self.current_epoch,
            n_samples=8,
            modalities=self.modalities,
            log_audio=log_audio,
        )
        if getattr(self.cfg.training, "log_rotation_matrices", True):
            log_rotation_matrices(self, experiment, self.current_epoch, self.modalities)
        if getattr(self.cfg.training, "log_modality_alignment", True):
            val_loader = self.trainer.datamodule.val_dataloader()
            log_modality_alignment(self, val_loader, experiment, self.current_epoch, self.modalities)
        log_tsne_every = getattr(self.cfg.training, "log_tsne_every_n_epochs", 0)
        if log_tsne_every > 0 and (self.current_epoch + 1) % log_tsne_every == 0:
            val_loader = self.trainer.datamodule.val_dataloader()
            log_tsne_embeddings(
                self, val_loader, experiment, self.current_epoch, self.modalities, max_samples=500
            )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.cfg.training.lr,
            weight_decay=self.cfg.training.weight_decay,
        )

        warmup_epochs = self.cfg.training.warmup_epochs
        max_epochs = self.cfg.training.max_epochs

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: _warmup_cosine_schedule(epoch, warmup_epochs, max_epochs),
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


def _warmup_cosine_schedule(epoch: int, warmup_epochs: int, max_epochs: int) -> float:
    """Linear warmup then cosine annealing to 0."""
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(max_epochs - warmup_epochs, 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))
