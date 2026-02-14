"""CREMA-D Dataset and DataModule for PyTorch Lightning.

CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset) contains 7,442
audio-visual clips from 91 actors expressing 6 emotions: anger, disgust,
fear, happiness, sadness, and neutral.

Download source: Original GitHub repo (CheyneyComputerScience/CREMA-D) via
git-lfs, which provides both AudioWAV/ and VideoFlash/ directories.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

log = logging.getLogger(__name__)

# Emotion label mapping (alphabetical order for determinism)
EMOTION_LABELS = {
    "ANG": 0,  # Anger
    "DIS": 1,  # Disgust
    "FEA": 2,  # Fear
    "HAP": 3,  # Happiness
    "NEU": 4,  # Neutral
    "SAD": 5,  # Sadness
}
NUM_CLASSES = len(EMOTION_LABELS)


def _parse_cremad_filename(stem: str) -> tuple[str, str, str, str]:
    """Parse CREMA-D filename like '1001_DFA_ANG_XX'.

    Returns:
        (actor_id, sentence_code, emotion_code, intensity_code)
    """
    parts = stem.split("_")
    if len(parts) != 4:
        raise ValueError(f"Unexpected CREMA-D filename format: {stem}")
    return parts[0], parts[1], parts[2], parts[3]


CREMAD_REPO_URL = "https://github.com/CheyneyComputerScience/CREMA-D.git"
CREMAD_MIRROR_URL = "https://gitlab.com/cs-cooper-lab/crema-d-mirror.git"


def download_cremad(data_dir: str | Path) -> Path:
    """Download CREMA-D from the original GitHub repo (requires git-lfs).

    The repo contains AudioWAV/ and VideoFlash/ directories with the
    paired audio and video files needed for multimodal training.

    Falls back to a GitLab mirror if the GitHub clone fails.

    Returns path to the cloned repository root.
    """
    import shutil
    import subprocess

    data_dir = Path(data_dir)
    cremad_raw = data_dir / "cremad_raw"

    # Check if already downloaded with actual audio+video content
    audio_dir = cremad_raw / "AudioWAV"
    video_dir = cremad_raw / "VideoFlash"
    if audio_dir.exists() and video_dir.exists():
        wav_count = len(list(audio_dir.glob("*.wav")))
        flv_count = len(list(video_dir.glob("*.flv")))
        if wav_count > 7000 and flv_count > 7000:
            log.info(
                "CREMA-D already downloaded at %s (%d audio, %d video), skipping.",
                cremad_raw,
                wav_count,
                flv_count,
            )
            return cremad_raw

    # Verify git-lfs is installed
    try:
        subprocess.run(
            ["git", "lfs", "version"],
            check=True,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError(
            "git-lfs is required to download CREMA-D video files. "
            "Install it with: brew install git-lfs && git lfs install "
            "(macOS) or apt install git-lfs && git lfs install (Linux)."
        )

    cremad_raw.mkdir(parents=True, exist_ok=True)

    # Try GitHub first, fall back to GitLab mirror
    for url in [CREMAD_REPO_URL, CREMAD_MIRROR_URL]:
        log.info("Cloning CREMA-D from %s (~7.5 GB, this may take a while) ...", url)
        try:
            subprocess.run(
                ["git", "clone", url, str(cremad_raw)],
                check=True,
            )
            log.info("CREMA-D download complete at %s", cremad_raw)
            return cremad_raw
        except subprocess.CalledProcessError as e:
            log.warning("Clone from %s failed: %s", url, e)
            # Clean up partial clone before trying mirror
            if cremad_raw.exists():
                shutil.rmtree(cremad_raw)
            cremad_raw.mkdir(parents=True, exist_ok=True)

    raise RuntimeError(
        "Failed to download CREMA-D from both GitHub and GitLab mirror. "
        "Check your internet connection and git-lfs installation."
    )


def _generate_spectrogram(
    wav_path: Path,
    target_size: tuple[int, int] = (112, 112),
    sr: int = 22050,
    n_mels: int = 128,
) -> np.ndarray:
    """Convert a single wav file to a mel spectrogram normalized to [0, 1]."""
    import librosa
    from scipy.ndimage import zoom

    y, _ = librosa.load(str(wav_path), sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    h, w = mel_db.shape
    zoom_factors = (target_size[0] / h, target_size[1] / w)
    mel_resized = zoom(mel_db, zoom_factors, order=1)

    mel_min = mel_resized.min()
    mel_max = mel_resized.max()
    if mel_max - mel_min > 0:
        mel_norm = (mel_resized - mel_min) / (mel_max - mel_min)
    else:
        mel_norm = np.zeros_like(mel_resized)

    return mel_norm.astype(np.float32)


def _extract_video_frame(
    video_path: Path,
    target_size: tuple[int, int] = (64, 64),
) -> np.ndarray:
    """Extract the center frame from a video and convert to grayscale.

    Returns a float32 array of shape (target_size[0], target_size[1])
    normalized to [0, 1].
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        center = max(total_frames // 2, 0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, center)
        ret, frame = cap.read()
        if not ret:
            # Fallback: try the first frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"Could not read any frame from {video_path}")
    finally:
        cap.release()

    # Convert to grayscale and resize
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (target_size[1], target_size[0]))
    return resized.astype(np.float32) / 255.0


def _discover_samples(raw_dir: Path) -> list[dict]:
    """Discover paired audio+video files from the CREMA-D clone.

    Looks for the canonical directory layout from the GitHub repo:
      - AudioWAV/  (wav files)
      - VideoFlash/ (flv files)

    Falls back to alternative names (audios/, videos/) if needed.

    Returns a list of dicts with keys: 'audio_path', 'video_path' (or None),
    'emotion', 'actor_id'.
    """
    # Try canonical GitHub repo layout first, then alternatives
    audio_dir = None
    for candidate in ["AudioWAV", "audios"]:
        d = raw_dir / candidate
        if d.exists() and any(d.glob("*.wav")):
            audio_dir = d
            break
    if audio_dir is None:
        # Last resort: wav files in the root
        if any(raw_dir.glob("*.wav")):
            audio_dir = raw_dir
        else:
            raise FileNotFoundError(
                f"No .wav files found in {raw_dir}. "
                f"Check that CREMA-D downloaded correctly."
            )

    video_dir = None
    for candidate in ["VideoFlash", "videos"]:
        d = raw_dir / candidate
        if d.exists() and (any(d.glob("*.flv")) or any(d.glob("*.mp4"))):
            video_dir = d
            break

    wav_files = sorted(audio_dir.glob("*.wav"))
    log.info(
        "Found %d audio files in %s, video dir: %s",
        len(wav_files),
        audio_dir.name,
        video_dir.name if video_dir else "NONE (audio-only)",
    )
    if video_dir is None:
        log.warning(
            "No video directory found! Vision frames will be blank. "
            "Make sure git-lfs is installed and the repo was cloned fully."
        )

    samples = []
    videos_found = 0
    for wav_path in wav_files:
        stem = wav_path.stem
        try:
            actor_id, _sentence, emotion_code, _intensity = _parse_cremad_filename(stem)
        except ValueError:
            log.warning("Skipping file with unexpected name: %s", stem)
            continue

        if emotion_code not in EMOTION_LABELS:
            log.warning("Skipping unknown emotion %s in %s", emotion_code, stem)
            continue

        video_path = None
        if video_dir is not None:
            for ext in [".flv", ".mp4", ".avi"]:
                candidate = video_dir / (stem + ext)
                if candidate.exists():
                    video_path = candidate
                    videos_found += 1
                    break

        samples.append({
            "audio_path": wav_path,
            "video_path": video_path,
            "emotion": emotion_code,
            "actor_id": actor_id,
        })

    log.info(
        "Discovered %d CREMA-D samples (%d with video, %d audio-only)",
        len(samples),
        videos_found,
        len(samples) - videos_found,
    )
    return samples


def prepare_cremad(
    data_dir: str | Path,
    vision_size: int = 64,
    seed: int = 42,
) -> Path:
    """Download and prepare CREMA-D into train/val/test numpy arrays.

    Creates:
      - {output_dir}/{split}/frames.npy     shape (N, 1, vision_size, vision_size)
      - {output_dir}/{split}/spectrograms.npy  shape (N, 1, 112, 112)
      - {output_dir}/{split}/labels.npy      shape (N,)

    When video files are not available (audio-only HF download), frames are
    generated as blank images and a warning is logged.

    Returns the output directory path.
    """
    data_dir = Path(data_dir)
    output_dir = data_dir / "cremad"

    # Check if already prepared
    if (output_dir / "train" / "spectrograms.npy").exists():
        log.info("CREMA-D already prepared at %s, skipping.", output_dir)
        return output_dir

    # Download
    raw_dir = download_cremad(data_dir)

    # Discover samples
    samples = _discover_samples(raw_dir)
    if not samples:
        raise RuntimeError("No valid CREMA-D samples found after discovery.")

    # Check that video files exist — training on blank frames is useless
    n_with_video = sum(1 for s in samples if s["video_path"] is not None)
    if n_with_video == 0:
        raise RuntimeError(
            "CREMA-D download has NO video files — all vision data would be blank. "
            "This usually means git-lfs was not installed when cloning. Fix:\n"
            "  1. Install git-lfs: brew install git-lfs && git lfs install\n"
            "  2. Delete the data: rm -rf data/cremad_raw data/cremad\n"
            "  3. Re-download: pavlov-download data=cremad"
        )
    elif n_with_video < len(samples):
        log.warning(
            "%d / %d samples are missing video files — those will use blank frames.",
            len(samples) - n_with_video,
            len(samples),
        )

    rng = np.random.default_rng(seed)

    # Extract features — track actor IDs alongside data for splitting
    log.info("Extracting spectrograms and frames for %d samples ...", len(samples))
    all_spectrograms = []
    all_frames = []
    all_labels = []
    all_actor_ids = []
    skipped = 0

    for i, sample in enumerate(samples):
        if (i + 1) % 500 == 0:
            log.info("  Processing %d / %d ...", i + 1, len(samples))

        try:
            spec = _generate_spectrogram(sample["audio_path"])
        except Exception as e:
            log.warning("Failed spectrogram for %s: %s", sample["audio_path"].name, e)
            skipped += 1
            continue

        if sample["video_path"] is not None:
            try:
                frame = _extract_video_frame(
                    sample["video_path"],
                    target_size=(vision_size, vision_size),
                )
            except Exception as e:
                log.warning(
                    "Failed frame extraction for %s: %s",
                    sample["video_path"].name,
                    e,
                )
                # Use blank frame as fallback
                frame = np.zeros((vision_size, vision_size), dtype=np.float32)
        else:
            # Audio-only download — use blank frame placeholder
            frame = np.zeros((vision_size, vision_size), dtype=np.float32)

        all_spectrograms.append(spec[np.newaxis, :, :])  # (1, 112, 112)
        all_frames.append(frame[np.newaxis, :, :])  # (1, H, W)
        all_labels.append(EMOTION_LABELS[sample["emotion"]])
        all_actor_ids.append(sample["actor_id"])

    if skipped > 0:
        log.warning("Skipped %d samples due to processing errors.", skipped)

    spectrograms = np.stack(all_spectrograms, axis=0).astype(np.float32)
    frames = np.stack(all_frames, axis=0).astype(np.float32)
    labels = np.array(all_labels, dtype=np.int64)

    # Split by actor (80/10/10) to avoid data leakage
    unique_actors = sorted(set(all_actor_ids))
    rng.shuffle(unique_actors)
    n_actors = len(unique_actors)
    n_train = int(0.8 * n_actors)
    n_val = int(0.1 * n_actors)

    train_actors = set(unique_actors[:n_train])
    val_actors = set(unique_actors[n_train:n_train + n_val])
    # test_actors = everything else

    splits: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for i, actor_id in enumerate(all_actor_ids):
        if actor_id in train_actors:
            splits["train"].append(i)
        elif actor_id in val_actors:
            splits["val"].append(i)
        else:
            splits["test"].append(i)

    for split_name, indices in splits.items():
        idx_arr = np.array(indices)
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        np.save(split_dir / "frames.npy", frames[idx_arr])
        np.save(split_dir / "spectrograms.npy", spectrograms[idx_arr])
        np.save(split_dir / "labels.npy", labels[idx_arr])
        log.info("Saved %s split: %d samples", split_name, len(idx_arr))

    log.info("CREMA-D preparation complete. Output: %s", output_dir)
    return output_dir


class CREMADDataset(Dataset):
    """CREMA-D audio-visual emotion dataset.

    Loads pre-processed numpy arrays and returns samples as dicts with
    'vision', 'audio', and 'label' keys (same interface as AVMNISTDataset).
    """

    def __init__(self, split_dir: str | Path) -> None:
        split_dir = Path(split_dir)
        self.frames = np.load(split_dir / "frames.npy")
        self.spectrograms = np.load(split_dir / "spectrograms.npy")
        self.labels = np.load(split_dir / "labels.npy")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "vision": torch.from_numpy(self.frames[idx].copy()),
            "audio": torch.from_numpy(self.spectrograms[idx].copy()),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class CREMADDataModule(pl.LightningDataModule):
    """Lightning DataModule for CREMA-D.

    Handles downloading, preparation, and dataloader creation.
    Each batch is a dict: {'vision': (B,1,64,64), 'audio': (B,1,112,112), 'label': (B,)}.
    """

    def __init__(
        self,
        data_dir: str = "data",
        batch_size: int = 64,
        num_workers: int = 4,
        vision_size: int = 64,
        **kwargs,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.vision_size = vision_size
        self.train_dataset: CREMADDataset | None = None
        self.val_dataset: CREMADDataset | None = None
        self.test_dataset: CREMADDataset | None = None

    def prepare_data(self) -> None:
        """Download and prepare data (called on rank 0 only)."""
        cremad_dir = self.data_dir / "cremad"
        if not (cremad_dir / "train" / "spectrograms.npy").exists():
            prepare_cremad(str(self.data_dir), vision_size=self.vision_size)

    def setup(self, stage: str | None = None) -> None:
        """Load datasets for the given stage."""
        cremad_dir = self.data_dir / "cremad"
        if stage == "fit" or stage is None:
            self.train_dataset = CREMADDataset(cremad_dir / "train")
            self.val_dataset = CREMADDataset(cremad_dir / "val")
        if stage == "test" or stage is None:
            self.test_dataset = CREMADDataset(cremad_dir / "test")

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
