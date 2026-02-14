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
        self.images = np.load(split_dir / "images.npy", mmap_mode="r")
        self.spectrograms = np.load(split_dir / "spectrograms.npy", mmap_mode="r")
        self.labels = np.load(split_dir / "labels.npy")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "vision": torch.from_numpy(self.images[idx].copy()),
            "audio": torch.from_numpy(self.spectrograms[idx].copy()),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
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

    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_dataset is not None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        assert self.test_dataset is not None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
