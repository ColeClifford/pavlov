"""Download and prepare AV-MNIST dataset.

Downloads MNIST via torchvision and FSDD from GitHub, generates mel spectrograms,
and pairs audio-visual samples by digit label.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torchvision

log = logging.getLogger(__name__)


def download_mnist(data_dir: str | Path) -> Path:
    """Download MNIST using torchvision. Returns path to MNIST root."""
    data_dir = Path(data_dir)
    mnist_dir = data_dir / "mnist"
    if mnist_dir.exists() and any(mnist_dir.iterdir()):
        log.info("MNIST already downloaded at %s, skipping.", mnist_dir)
        return mnist_dir
    log.info("Downloading MNIST to %s ...", mnist_dir)
    torchvision.datasets.MNIST(root=str(mnist_dir), train=True, download=True)
    torchvision.datasets.MNIST(root=str(mnist_dir), train=False, download=True)
    return mnist_dir


FSDD_REPO_URL = "https://github.com/Jakobovski/free-spoken-digit-dataset.git"


def download_fsdd(data_dir: str | Path) -> Path:
    """Clone FSDD repo to get wav recordings. Returns path to recordings dir."""
    data_dir = Path(data_dir)
    recordings_dir = data_dir / "fsdd" / "recordings"
    if recordings_dir.exists() and any(recordings_dir.glob("*.wav")):
        log.info("FSDD already downloaded at %s, skipping.", recordings_dir)
        return recordings_dir
    log.info("Cloning FSDD repository ...")
    fsdd_dir = data_dir / "fsdd"
    fsdd_dir.mkdir(parents=True, exist_ok=True)
    # Shallow clone to save bandwidth — we only need the recordings
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            ["git", "clone", "--depth", "1", FSDD_REPO_URL, tmp],
            check=True,
        )
        src = Path(tmp) / "recordings"
        if recordings_dir.exists():
            shutil.rmtree(recordings_dir)
        shutil.copytree(src, recordings_dir)
    log.info("FSDD recordings saved to %s", recordings_dir)
    return recordings_dir


def generate_spectrograms(
    recordings_dir: str | Path,
    output_dir: str | Path,
    target_size: tuple[int, int] = (112, 112),
    sr: int = 22050,
    n_mels: int = 128,
) -> Path:
    """Convert FSDD wav files to mel spectrograms saved as .npy files.

    Each spectrogram is normalized to [0, 1] and resized to *target_size*.
    Output files are named identically to inputs but with .npy extension.
    """
    import librosa
    from scipy.ndimage import zoom

    recordings_dir = Path(recordings_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(recordings_dir.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No wav files found in {recordings_dir}")

    # Check if spectrograms already generated
    existing = list(output_dir.glob("*.npy"))
    if len(existing) >= len(wav_files):
        log.info("Spectrograms already generated (%d files), skipping.", len(existing))
        return output_dir

    log.info("Generating %d spectrograms ...", len(wav_files))
    for wav_path in wav_files:
        y, _ = librosa.load(str(wav_path), sr=sr)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Resize to target_size
        h, w = mel_db.shape
        zoom_factors = (target_size[0] / h, target_size[1] / w)
        mel_resized = zoom(mel_db, zoom_factors, order=1)

        # Normalize to [0, 1]
        mel_min = mel_resized.min()
        mel_max = mel_resized.max()
        if mel_max - mel_min > 0:
            mel_norm = (mel_resized - mel_min) / (mel_max - mel_min)
        else:
            mel_norm = np.zeros_like(mel_resized)

        out_path = output_dir / (wav_path.stem + ".npy")
        np.save(out_path, mel_norm.astype(np.float32))

    log.info("Spectrograms saved to %s", output_dir)
    return output_dir


def _parse_fsdd_filename(stem: str) -> tuple[int, str, int]:
    """Parse FSDD filename stem like '0_jackson_0' -> (digit, speaker, index)."""
    parts = stem.split("_")
    digit = int(parts[0])
    speaker = "_".join(parts[1:-1])  # handle speaker names with underscores
    index = int(parts[-1])
    return digit, speaker, index


def pair_avmnist(
    mnist_dir: str | Path,
    spec_dir: str | Path,
    output_dir: str | Path,
    seed: int = 42,
) -> Path:
    """Pair MNIST images with FSDD spectrograms by digit label.

    Creates train/val/test splits (0.8/0.1/0.1) and saves:
      - {split}/images.npy    shape (N, 1, 28, 28)
      - {split}/spectrograms.npy  shape (N, 1, 112, 112)
      - {split}/labels.npy    shape (N,)
    """
    mnist_dir = Path(mnist_dir)
    spec_dir = Path(spec_dir)
    output_dir = Path(output_dir)

    # Check if already paired
    if (output_dir / "train" / "images.npy").exists():
        log.info("Paired AV-MNIST already exists at %s, skipping.", output_dir)
        return output_dir

    rng = np.random.default_rng(seed)

    # Load MNIST
    mnist_train = torchvision.datasets.MNIST(
        root=str(mnist_dir), train=True, download=False
    )
    mnist_test = torchvision.datasets.MNIST(
        root=str(mnist_dir), train=False, download=False
    )

    # Combine all MNIST data
    all_images = np.concatenate(
        [mnist_train.data.numpy(), mnist_test.data.numpy()], axis=0
    )
    all_labels = np.concatenate(
        [mnist_train.targets.numpy(), mnist_test.targets.numpy()], axis=0
    )

    # Group spectrograms by digit
    spec_by_digit: dict[int, list[np.ndarray]] = {d: [] for d in range(10)}
    for spec_path in sorted(spec_dir.glob("*.npy")):
        digit, _, _ = _parse_fsdd_filename(spec_path.stem)
        spec_by_digit[digit].append(np.load(spec_path))

    for d in range(10):
        if not spec_by_digit[d]:
            raise ValueError(f"No spectrograms found for digit {d}")

    # Pair each MNIST image with a random spectrogram of the same digit
    n = len(all_images)
    images = all_images[:, np.newaxis, :, :].astype(np.float32) / 255.0  # (N,1,28,28)
    spectrograms = np.zeros((n, 1, 112, 112), dtype=np.float32)
    for i in range(n):
        digit = int(all_labels[i])
        specs = spec_by_digit[digit]
        chosen = specs[rng.integers(len(specs))]
        spectrograms[i, 0] = chosen

    labels = all_labels.astype(np.int64)

    # Shuffle and split
    indices = rng.permutation(n)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)

    splits = {
        "train": indices[:n_train],
        "val": indices[n_train : n_train + n_val],
        "test": indices[n_train + n_val :],
    }

    for split_name, idx in splits.items():
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        np.save(split_dir / "images.npy", images[idx])
        np.save(split_dir / "spectrograms.npy", spectrograms[idx])
        np.save(split_dir / "labels.npy", labels[idx])
        log.info("Saved %s split: %d samples", split_name, len(idx))

    return output_dir


def download_and_prepare(data_dir: str | Path) -> Path:
    """Orchestrate full AV-MNIST download and preparation pipeline.

    Idempotent: skips steps where output files already exist.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    log.info("Preparing AV-MNIST dataset in %s", data_dir)

    mnist_dir = download_mnist(data_dir)
    recordings_dir = download_fsdd(data_dir)
    spec_dir = generate_spectrograms(recordings_dir, data_dir / "fsdd" / "spectrograms")
    output_dir = pair_avmnist(mnist_dir, spec_dir, data_dir / "avmnist")

    log.info("AV-MNIST preparation complete. Output: %s", output_dir)
    return output_dir
