"""Training entry point for Pavlov."""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint

log = logging.getLogger(__name__)

# Resolve config path relative to package root (works when run via entry point)
_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "configs"


@hydra.main(version_base=None, config_path=str(_CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed)

    from pavlov.data import build_datamodule
    from pavlov.models.lightning_module import PavlovLightningModule

    datamodule = build_datamodule(cfg)
    model = PavlovLightningModule(cfg)

    # Optional torch.compile for PyTorch 2.0+ graph optimizations
    if getattr(cfg.training, "compile_model", False):
        if hasattr(torch, "compile"):
            log.info("Compiling model with torch.compile()")
            model.model = torch.compile(model.model)
        else:
            log.warning("torch.compile not available (requires PyTorch 2.0+), skipping")

    # Use Hydra's output dir so logs/checkpoints live with config in outputs/YYYY-MM-DD/HH-MM-SS/
    output_dir = HydraConfig.get().runtime.output_dir

    # Logger selection
    if cfg.training.logger == "wandb":
        from pytorch_lightning.loggers import WandbLogger

        logger = WandbLogger(
            project=cfg.training.wandb.project,
            entity=cfg.training.wandb.entity,
            save_dir=output_dir,
        )
    else:
        from pytorch_lightning.loggers import TensorBoardLogger

        logger = TensorBoardLogger(save_dir=output_dir, name="logs")

    callbacks = [
        ModelCheckpoint(
            monitor="val/total_loss",
            mode="min",
            save_top_k=3,
            filename="pavlov-{epoch:02d}-{val/total_loss:.4f}",
        ),
    ]

    # Read precision and gradient accumulation from config
    precision = getattr(cfg.training, "precision", "32-true")
    accumulate_grad_batches = getattr(cfg.training, "accumulate_grad_batches", 1)

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        logger=logger,
        callbacks=callbacks,
        accelerator="auto",
        devices=1,
        precision=precision,
        accumulate_grad_batches=accumulate_grad_batches,
    )
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    main()
