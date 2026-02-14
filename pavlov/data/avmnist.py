"""AV-MNIST Dataset and DataModule for PyTorch Lightning."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from pavlov.data.download import download_and_prepare


class AVMNISTDataset(Dataset):
    """Audio-visual MNIST dataset.

    Loads pre-processed numpy arrays and returns samples as dicts with
    'vision', 'audio', and 'label' keys.
    """

    def __init__(self, split_dir: str | Path) -> None:
        split_dir = Path(split_dir)
        # Load fully into memory (dataset is ~2.5 GB total) to avoid
        # mmap per-sample I/O overhead and enable zero-copy tensor creation.
        self.images = torch.from_numpy(np.load(split_dir / "images.npy"))
        self.spectrograms = torch.from_numpy(np.load(split_dir / "spectrograms.npy"))
        self.labels = torch.from_numpy(np.load(split_dir / "labels.npy")).long()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "vision": self.images[idx],
            "audio": self.spectrograms[idx],
            "label": self.labels[idx],
        }


class AVMNISTDataModule(pl.LightningDataModule):
    """Lightning DataModule for AV-MNIST.

    Handles downloading, preparation, and dataloader creation.
    Each batch is a dict: {'vision': (B,1,28,28), 'audio': (B,1,112,112), 'label': (B,)}.
    """

    def __init__(
        self,
        data_dir: str = "data",
        batch_size: int = 64,
        num_workers: int = 4,
        **kwargs,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset: AVMNISTDataset | None = None
        self.val_dataset: AVMNISTDataset | None = None
        self.test_dataset: AVMNISTDataset | None = None

    def prepare_data(self) -> None:
        """Download and prepare data (called on rank 0 only)."""
        avmnist_dir = self.data_dir / "avmnist"
        if not (avmnist_dir / "train" / "images.npy").exists():
            download_and_prepare(str(self.data_dir))

    def setup(self, stage: str | None = None) -> None:
        """Load datasets for the given stage."""
        avmnist_dir = self.data_dir / "avmnist"
        if stage == "fit" or stage is None:
            self.train_dataset = AVMNISTDataset(avmnist_dir / "train")
            self.val_dataset = AVMNISTDataset(avmnist_dir / "val")
        if stage == "test" or stage is None:
            self.test_dataset = AVMNISTDataset(avmnist_dir / "test")

    def _loader_kwargs(self) -> dict:
        """Common DataLoader keyword arguments for performance."""
        kwargs: dict = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": True,
        }
        if self.num_workers > 0:
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"] = 4
        return kwargs

    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            **self._loader_kwargs(),
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_dataset is not None
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            **self._loader_kwargs(),
        )

    def test_dataloader(self) -> DataLoader:
        assert self.test_dataset is not None
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            **self._loader_kwargs(),
        )
