"""Evaluation entry point for Pavlov."""

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

    from pavlov.data.avmnist import AVMNISTDataModule
    from pavlov.evaluation.cross_modal_transfer import evaluate_cross_modal_transfer
    from pavlov.evaluation.retrieval import evaluate_retrieval
    from pavlov.evaluation.sample_logging import log_samples_from_dataloader
    from pavlov.models.lightning_module import PavlovLightningModule

    # Load from checkpoint
    if not hasattr(cfg, "checkpoint") or cfg.checkpoint is None:
        raise ValueError("Must specify +checkpoint=<path> to a model checkpoint.")

    model = PavlovLightningModule.load_from_checkpoint(
        cfg.checkpoint, weights_only=False
    )
    model.eval()

    datamodule = AVMNISTDataModule(**cfg.data)
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

    print("=" * 60)
    print("Pavlov Evaluation Results")
    print("=" * 60)

    # Cross-modal transfer
    for train_mod in modalities:
        for test_mod in modalities:
            if train_mod == test_mod:
                continue
            results = evaluate_cross_modal_transfer(
                model, test_loader, train_mod, test_mod
            )
            print(f"\nCross-modal transfer: {train_mod} -> {test_mod}")
            print(f"  Train accuracy:    {results['train_accuracy']:.4f}")
            print(f"  Transfer accuracy: {results['transfer_accuracy']:.4f}")

    # Cross-modal retrieval
    for query_mod in modalities:
        for gallery_mod in modalities:
            if query_mod == gallery_mod:
                continue
            results = evaluate_retrieval(
                model, test_loader, query_mod, gallery_mod, k=10
            )
            print(f"\nRetrieval: {query_mod} -> {gallery_mod}")
            for key, val in results.items():
                print(f"  {key}: {val:.4f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
