"""Visualization entry point for Pavlov."""

from __future__ import annotations

from pathlib import Path

import hydra
import pytorch_lightning as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

# Resolve config path relative to package root (works when run via entry point)
_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "configs"


@hydra.main(version_base=None, config_path=str(_CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed)

    from pavlov.data import build_datamodule
    from pavlov.evaluation.sample_logging import log_samples_from_dataloader
    from pavlov.evaluation.visualization import plot_embeddings
    from pavlov.models.lightning_module import PavlovLightningModule
    from pavlov.utils.checkpoint import load_model_from_checkpoint

    if not hasattr(cfg, "checkpoint") or cfg.checkpoint is None:
        raise ValueError("Must specify +checkpoint=<path> to a model checkpoint.")

    model = load_model_from_checkpoint(
        PavlovLightningModule,
        cfg.checkpoint,
        weights_only=False,
    )
    model.eval()

    datamodule = build_datamodule(cfg)
    datamodule.setup("test")
    test_loader = datamodule.test_dataloader()

    modalities = list(cfg.modalities)

    # Optional: log sample reconstructions to TensorBoard
    if cfg.get("log_tensorboard", False):
        from torch.utils.tensorboard import SummaryWriter

        log_dir = cfg.get("log_dir") or HydraConfig.get().runtime.output_dir
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        with SummaryWriter(log_dir=str(log_dir)) as writer:
            log_samples_from_dataloader(
                model, test_loader, writer, step=0, modalities=modalities
            )
        print(f"Logged sample reconstructions to TensorBoard at {log_dir}")

    save_path = cfg.get("save_path", "embeddings_tsne.png")

    plot_embeddings(
        model,
        test_loader,
        save_path=save_path,
        modalities=modalities,
    )


if __name__ == "__main__":
    main()
