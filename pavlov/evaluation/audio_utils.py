"""Audio utilities for spectrogram inversion (AV-MNIST pipeline)."""

from __future__ import annotations

import numpy as np
import torch

# Match pavlov/data/download.py generate_spectrograms
AVMNIST_SR = 22050
AVMNIST_N_MELS = 128
AVMNIST_TARGET_SIZE = (112, 112)


def spectrogram_to_audio(
    spec: np.ndarray | torch.Tensor,
    sr: int = AVMNIST_SR,
    n_mels: int = AVMNIST_N_MELS,
    n_fft: int = 2048,
    hop_length: int = 512,
    target_duration: float = 0.7,
    ref_db: float = 0.0,
    min_db: float = -80.0,
) -> np.ndarray:
    """Convert normalized mel spectrogram back to audio.

    Reverses the pipeline in pavlov/data/download.py:
    - Input: spectrogram in [0, 1] (normalized from dB), shape (1, H, W) or (H, W)
    - Denormalize to dB, convert to power, resize to (n_mels, n_frames), invert mel

    Args:
        spec: Spectrogram tensor/array, shape (1, 112, 112) or (112, 112).
        sr: Sample rate.
        n_mels: Number of mel bands for inversion.
        n_fft: FFT size (must match mel computation).
        hop_length: Hop length (must match mel computation).
        target_duration: Expected audio duration in seconds. The original FSDD
            spectrograms (~30-50 time frames for 0.5-1.0s clips) were stretched
            to 112 frames during dataset creation; this parameter controls the
            time-axis resize during inversion so the output has the right speed.
        ref_db: Reference dB for denormalization (1.0 maps to this).
        min_db: Minimum dB for denormalization (0.0 maps to this).

    Returns:
        Audio array, shape (n_samples,), values in [-1, 1].
    """
    import librosa
    from scipy.ndimage import zoom

    if isinstance(spec, torch.Tensor):
        spec = spec.cpu().numpy()

    # Remove channel dim if present
    if spec.ndim == 3:
        spec = spec.squeeze(0)

    # Denormalize [0, 1] -> dB range
    mel_db = min_db + (ref_db - min_db) * spec

    # dB -> power
    mel_power = librosa.db_to_power(mel_db)

    # Resize from (112, 112) to (n_mels, target_n_frames) for mel_to_audio.
    # The original FSDD spectrograms had ~30-50 time frames (0.5-1.0s clips)
    # that were stretched to 112 frames during dataset creation. We must
    # compress the time axis back to a realistic frame count, otherwise
    # mel_to_audio produces audio ~3x too long (slow-motion).
    h, w = mel_power.shape
    target_n_frames = max(1, int(target_duration * sr / hop_length))
    zoom_factors = (n_mels / h, target_n_frames / w)
    mel_resized = zoom(mel_power, zoom_factors, order=1)

    # Invert mel to audio
    y = librosa.feature.inverse.mel_to_audio(
        mel_resized,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
    )

    # Normalize to [-1, 1] for TensorBoard
    y_max = np.abs(y).max()
    if y_max > 0:
        y = y / y_max
    return y.astype(np.float32)
