import torch


def skew_symmetric_to_rotation(params: torch.Tensor, dim: int) -> torch.Tensor:
    """Convert a vector of d(d-1)/2 parameters to a rotation matrix in SO(d).

    Constructs a skew-symmetric matrix A from the parameter vector and returns
    the rotation matrix R = matrix_exp(A). This parameterization guarantees R
    is always a valid rotation matrix (orthogonal with determinant +1).

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
    idx = 0
    for i in range(dim):
        for j in range(i + 1, dim):
            A[i, j] = params[idx]
            A[j, i] = -params[idx]
            idx += 1

    return torch.linalg.matrix_exp(A)
