from __future__ import annotations

import pytorch_lightning as pl
from omegaconf import DictConfig

from pavlov.data.avmnist import AVMNISTDataset, AVMNISTDataModule


def build_datamodule(cfg: DictConfig) -> pl.LightningDataModule:
    """Factory function to create a dataset DataModule from config.

    Dispatches based on ``cfg.data.dataset`` (defaults to ``"avmnist"``
    when the field is absent for backward compatibility).

    Args:
        cfg: Full Hydra config. Must contain a ``data`` sub-config.

    Returns:
        A LightningDataModule instance.
    """
    dataset_name = cfg.data.get("dataset", "avmnist")

    if dataset_name == "avmnist":
        return AVMNISTDataModule(**cfg.data)
    elif dataset_name == "cremad":
        from pavlov.data.cremad import CREMADDataModule
        return CREMADDataModule(**cfg.data)
    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name!r}. "
            f"Available: 'avmnist', 'cremad'"
        )


__all__ = ["AVMNISTDataset", "AVMNISTDataModule", "build_datamodule"]
