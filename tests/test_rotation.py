import torch
import pytest

from pavlov.utils.rotation import skew_symmetric_to_rotation


class TestSkewSymmetricToRotation:
    """Tests for the SO(d) rotation matrix parameterization."""

    def test_orthogonality_3d(self):
        """R^T @ R should equal identity for 3D rotation."""
        params = torch.randn(3)  # 3*(3-1)/2 = 3
        R = skew_symmetric_to_rotation(params, dim=3)
        eye = torch.eye(3)
        assert torch.allclose(R.T @ R, eye, atol=1e-5)

    def test_orthogonality_8d(self):
        """R^T @ R should equal identity for 8D rotation."""
        params = torch.randn(28)  # 8*(8-1)/2 = 28
        R = skew_symmetric_to_rotation(params, dim=8)
        eye = torch.eye(8)
        assert torch.allclose(R.T @ R, eye, atol=1e-5)

    def test_determinant_positive_one(self):
        """det(R) should be +1 (proper rotation, not reflection)."""
        params = torch.randn(6)  # 4*(4-1)/2 = 6
        R = skew_symmetric_to_rotation(params, dim=4)
        assert torch.allclose(torch.det(R), torch.tensor(1.0), atol=1e-5)

    def test_identity_at_zero_params(self):
        """Zero parameters should produce the identity matrix."""
        params = torch.zeros(3)
        R = skew_symmetric_to_rotation(params, dim=3)
        eye = torch.eye(3)
        assert torch.allclose(R, eye, atol=1e-6)

    def test_gradient_flows(self):
        """Gradients should flow through the rotation parameterization."""
        params = torch.randn(3, requires_grad=True)
        R = skew_symmetric_to_rotation(params, dim=3)
        loss = R.sum()
        loss.backward()
        assert params.grad is not None
        assert not torch.all(params.grad == 0)

    def test_wrong_param_count_raises(self):
        """Should raise assertion error for wrong number of parameters."""
        params = torch.randn(5)  # wrong for dim=3 (expects 3)
        with pytest.raises(AssertionError):
            skew_symmetric_to_rotation(params, dim=3)

    def test_2d_rotation(self):
        """2D rotation should be a simple rotation matrix."""
        angle = torch.tensor([0.5])  # 2*(2-1)/2 = 1 param
        R = skew_symmetric_to_rotation(angle, dim=2)
        # For 2D, A = [[0, 0.5], [-0.5, 0]], exp(A) = [[cos, sin], [-sin, cos]]
        assert torch.allclose(R.T @ R, torch.eye(2), atol=1e-5)
        assert torch.allclose(torch.det(R), torch.tensor(1.0), atol=1e-5)
