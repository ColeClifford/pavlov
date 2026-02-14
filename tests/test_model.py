import torch
import pytest
from types import SimpleNamespace

from pavlov.models.pavlov_model import PavlovModel


@pytest.fixture
def model():
    """Create a PavlovModel with default test config."""
    cfg = SimpleNamespace(embed_dim=128)
    return PavlovModel(
        modalities=["vision", "audio"],
        embed_dim=128,
        latent_dim=32,
        model_cfg=cfg,
    )


class TestPavlovModel:
    """Tests for the PavlovModel end-to-end."""

    def test_encode_vision_shape(self, model):
        """Vision encoding should produce (batch, latent_dim) output."""
        x = torch.randn(4, 1, 28, 28)
        z = model.encode(x, "vision")
        assert z.shape == (4, 32)

    def test_encode_audio_shape(self, model):
        """Audio encoding should produce (batch, latent_dim) output."""
        x = torch.randn(4, 1, 112, 112)
        z = model.encode(x, "audio")
        assert z.shape == (4, 32)

    def test_decode_vision_shape(self, model):
        """Vision decoding should produce (batch, 1, 28, 28) output."""
        z = torch.randn(4, 32)
        recon = model.decode(z, "vision")
        assert recon.shape == (4, 1, 28, 28)

    def test_decode_audio_shape(self, model):
        """Audio decoding should produce (batch, 1, 112, 112) output."""
        z = torch.randn(4, 32)
        recon = model.decode(z, "audio")
        assert recon.shape == (4, 1, 112, 112)

    def test_cross_modal_decode(self, model):
        """Encode vision, decode as audio (cross-modal transfer)."""
        x_vision = torch.randn(4, 1, 28, 28)
        z = model.encode(x_vision, "vision")
        recon_audio = model.decode(z, "audio")
        assert recon_audio.shape == (4, 1, 112, 112)

    def test_forward_full(self, model):
        """Full forward pass should produce reconstructions and latents."""
        inputs = {
            "vision": torch.randn(4, 1, 28, 28),
            "audio": torch.randn(4, 1, 112, 112),
        }
        out = model(inputs)

        assert "reconstructions" in out
        assert "z" in out

        # Check latent shapes
        assert out["z"]["vision"].shape == (4, 32)
        assert out["z"]["audio"].shape == (4, 32)

        # Check reconstruction shapes (source -> target)
        assert out["reconstructions"]["vision"]["vision"].shape == (4, 1, 28, 28)
        assert out["reconstructions"]["vision"]["audio"].shape == (4, 1, 112, 112)
        assert out["reconstructions"]["audio"]["vision"].shape == (4, 1, 28, 28)
        assert out["reconstructions"]["audio"]["audio"].shape == (4, 1, 112, 112)

    def test_rotation_is_valid(self, model):
        """Each modality rotation should be in SO(d)."""
        for modality in model.modalities:
            R = model.get_rotation(modality)
            eye = torch.eye(model.latent_dim)
            assert torch.allclose(R.T @ R, eye, atol=1e-5)
            assert torch.allclose(torch.det(R), torch.tensor(1.0), atol=1e-5)

    def test_gradients_flow(self, model):
        """Gradients should flow through the full encode-decode pipeline."""
        x = torch.randn(2, 1, 28, 28)
        z = model.encode(x, "vision")
        recon = model.decode(z, "vision")
        loss = recon.sum()
        loss.backward()

        # Check rotation params get gradients
        assert model.rotation_params["vision"].grad is not None
        # Check shared projection gets gradients
        assert model.W.weight.grad is not None
        assert model.W_dec.weight.grad is not None
