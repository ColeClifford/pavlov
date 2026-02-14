"""Tests for the AV-MNIST data pipeline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from pavlov.data.avmnist import AVMNISTDataset, AVMNISTDataModule
from pavlov.data.download import _parse_fsdd_filename


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_fake_split(split_dir: Path, n: int = 32) -> None:
    """Create fake .npy files for a single split."""
    split_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    images = rng.random((n, 1, 28, 28), dtype=np.float32)
    spectrograms = rng.random((n, 1, 112, 112), dtype=np.float32)
    labels = rng.integers(0, 10, size=(n,)).astype(np.int64)
    np.save(split_dir / "images.npy", images)
    np.save(split_dir / "spectrograms.npy", spectrograms)
    np.save(split_dir / "labels.npy", labels)


def _create_fake_avmnist(data_dir: Path, n_train: int = 32, n_val: int = 8, n_test: int = 8) -> None:
    """Create a complete fake AV-MNIST dataset."""
    avmnist_dir = data_dir / "avmnist"
    _create_fake_split(avmnist_dir / "train", n_train)
    _create_fake_split(avmnist_dir / "val", n_val)
    _create_fake_split(avmnist_dir / "test", n_test)


# ---------------------------------------------------------------------------
# Tests: filename parsing
# ---------------------------------------------------------------------------

class TestFSDDParsing:
    def test_standard_name(self):
        digit, speaker, idx = _parse_fsdd_filename("3_jackson_12")
        assert digit == 3
        assert speaker == "jackson"
        assert idx == 12

    def test_digit_zero(self):
        digit, speaker, idx = _parse_fsdd_filename("0_theo_0")
        assert digit == 0
        assert speaker == "theo"
        assert idx == 0

    def test_digit_nine(self):
        digit, speaker, idx = _parse_fsdd_filename("9_nicolas_49")
        assert digit == 9
        assert speaker == "nicolas"
        assert idx == 49


# ---------------------------------------------------------------------------
# Tests: AVMNISTDataset
# ---------------------------------------------------------------------------

class TestAVMNISTDataset:
    @pytest.fixture()
    def dataset(self, tmp_path: Path) -> AVMNISTDataset:
        _create_fake_split(tmp_path / "split", n=16)
        return AVMNISTDataset(tmp_path / "split")

    def test_length(self, dataset: AVMNISTDataset):
        assert len(dataset) == 16

    def test_getitem_keys(self, dataset: AVMNISTDataset):
        sample = dataset[0]
        assert set(sample.keys()) == {"vision", "audio", "label"}

    def test_vision_shape(self, dataset: AVMNISTDataset):
        sample = dataset[0]
        assert sample["vision"].shape == (1, 28, 28)

    def test_audio_shape(self, dataset: AVMNISTDataset):
        sample = dataset[0]
        assert sample["audio"].shape == (1, 112, 112)

    def test_label_dtype(self, dataset: AVMNISTDataset):
        sample = dataset[0]
        assert sample["label"].dtype == torch.long

    def test_vision_dtype(self, dataset: AVMNISTDataset):
        sample = dataset[0]
        assert sample["vision"].dtype == torch.float32

    def test_audio_dtype(self, dataset: AVMNISTDataset):
        sample = dataset[0]
        assert sample["audio"].dtype == torch.float32


# ---------------------------------------------------------------------------
# Tests: AVMNISTDataModule
# ---------------------------------------------------------------------------

class TestAVMNISTDataModule:
    @pytest.fixture()
    def datamodule(self, tmp_path: Path) -> AVMNISTDataModule:
        _create_fake_avmnist(tmp_path, n_train=32, n_val=8, n_test=8)
        dm = AVMNISTDataModule(data_dir=str(tmp_path), batch_size=4, num_workers=0)
        dm.setup()
        return dm

    def test_train_dataset_loaded(self, datamodule: AVMNISTDataModule):
        assert datamodule.train_dataset is not None
        assert len(datamodule.train_dataset) == 32

    def test_val_dataset_loaded(self, datamodule: AVMNISTDataModule):
        assert datamodule.val_dataset is not None
        assert len(datamodule.val_dataset) == 8

    def test_test_dataset_loaded(self, datamodule: AVMNISTDataModule):
        assert datamodule.test_dataset is not None
        assert len(datamodule.test_dataset) == 8

    def test_train_batch_format(self, datamodule: AVMNISTDataModule):
        batch = next(iter(datamodule.train_dataloader()))
        assert batch["vision"].shape == (4, 1, 28, 28)
        assert batch["audio"].shape == (4, 1, 112, 112)
        assert batch["label"].shape == (4,)

    def test_val_batch_format(self, datamodule: AVMNISTDataModule):
        batch = next(iter(datamodule.val_dataloader()))
        assert batch["vision"].shape == (4, 1, 28, 28)
        assert batch["audio"].shape == (4, 1, 112, 112)
        assert batch["label"].shape == (4,)

    def test_label_range(self, datamodule: AVMNISTDataModule):
        batch = next(iter(datamodule.train_dataloader()))
        assert batch["label"].min() >= 0
        assert batch["label"].max() <= 9

    def test_prepare_data_skips_if_exists(self, tmp_path: Path):
        """prepare_data should not call download if data already exists."""
        _create_fake_avmnist(tmp_path)
        dm = AVMNISTDataModule(data_dir=str(tmp_path), batch_size=4, num_workers=0)
        with patch("pavlov.data.avmnist.download_and_prepare") as mock_dl:
            dm.prepare_data()
            mock_dl.assert_not_called()
