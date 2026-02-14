from __future__ import annotations

from functools import lru_cache

import torch


@lru_cache(maxsize=16)
def _triu_indices(dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Cached upper-triangular indices (CPU) to avoid recomputing each call."""
    return torch.triu_indices(dim, dim, offset=1)


def skew_symmetric_to_rotation(params: torch.Tensor, dim: int) -> torch.Tensor:
    """Convert a vector of d(d-1)/2 parameters to a rotation matrix in SO(d).

    Constructs a skew-symmetric matrix A from the parameter vector and returns
    the rotation matrix R = matrix_exp(A). This parameterization guarantees R
    is always a valid rotation matrix (orthogonal with determinant +1).

    Uses vectorized index assignment instead of a Python loop for performance
    (e.g. ~8128 iterations avoided for dim=128).

    Args:
        params: Vector of shape (d*(d-1)//2,) -- the free parameters.
        dim: The dimension d of the rotation matrix.

    Returns:
        Rotation matrix of shape (d, d) in SO(d).
    """
    expected = dim * (dim - 1) // 2
    assert params.shape == (expected,), (
        f"Expected {expected} params for dim={dim}, got {params.shape}"
    )

    A = torch.zeros(dim, dim, dtype=params.dtype, device=params.device)
    rows, cols = _triu_indices(dim)
    rows = rows.to(params.device)
    cols = cols.to(params.device)
    A[rows, cols] = params
    A[cols, rows] = -params

    return torch.linalg.matrix_exp(A)
