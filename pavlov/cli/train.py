"""Training entry point for Pavlov."""

from __future__ import annotations

from pathlib import Path

import hydra
import pytorch_lightning as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint

# Resolve config path relative to package root (works when run via entry point)
_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "configs"


@hydra.main(version_base=None, config_path=str(_CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed)

    from pavlov.data.avmnist import AVMNISTDataModule
    from pavlov.models.lightning_module import PavlovLightningModule

    datamodule = AVMNISTDataModule(**cfg.data)
    model = PavlovLightningModule(cfg)

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

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        logger=logger,
        callbacks=callbacks,
        accelerator="auto",
        devices=1,
    )
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    main()
