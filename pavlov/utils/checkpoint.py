"""Checkpoint loading utilities."""

from __future__ import annotations

import io
from pathlib import Path

import torch


def load_checkpoint_for_eval(
    checkpoint_path: str | Path,
    map_location: str | torch.device | None = None,
    weights_only: bool = False,
) -> dict:
    """Load checkpoint and remap torch.compile _orig_mod keys for evaluation.

    When training with torch.compile(), the saved state_dict uses keys like
    model._orig_mod.W.weight. This remaps them to model.W.weight so the
    uncompiled model can load the weights.

    Returns:
        Checkpoint dict with remapped state_dict (if needed).
    """
    ckpt = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=weights_only,
    )
    if "state_dict" not in ckpt:
        return ckpt

    state_dict = ckpt["state_dict"]
    if not any("_orig_mod" in k for k in state_dict):
        return ckpt

    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("._orig_mod.", ".")
        new_state_dict[new_key] = v
    ckpt["state_dict"] = new_state_dict
    return ckpt


def load_model_from_checkpoint(
    model_class,
    checkpoint_path: str | Path,
    map_location: str | torch.device | None = None,
    weights_only: bool = False,
    **kwargs,
):
    """Load a LightningModule from checkpoint, handling torch.compile state_dict.

    Uses load_checkpoint_for_eval to remap keys, then loads via the model class.
    """
    ckpt = load_checkpoint_for_eval(checkpoint_path, map_location, weights_only)
    buffer = io.BytesIO()
    torch.save(ckpt, buffer)
    buffer.seek(0)
    return model_class.load_from_checkpoint(
        buffer,
        map_location=map_location,
        weights_only=weights_only,
        **kwargs,
    )
