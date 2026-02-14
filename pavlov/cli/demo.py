"""Interactive Gradio demo for Pavlov multimodal reconstructions."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

matplotlib.use("Agg")


def find_best_checkpoint(outputs_dir: str = "outputs") -> Path:
    """Find the checkpoint with the lowest total_loss in outputs/."""
    ckpts = list(Path(outputs_dir).rglob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(
            f"No .ckpt files found under {outputs_dir}/. "
            "Pass --checkpoint explicitly."
        )

    def parse_loss(p: Path) -> float:
        name = p.stem
        for part in name.split("-"):
            if part.startswith("total_loss="):
                try:
                    return float(part.split("=")[1])
                except ValueError:
                    pass
        return float("inf")

    best = min(ckpts, key=parse_loss)
    loss = parse_loss(best)
    print(f"Auto-selected checkpoint: {best} (total_loss={loss:.4f})")
    return best


def load_demo_model(checkpoint_path: str | Path, data_dir: str = "data"):
    """Load model and test dataset for the demo."""
    from pavlov.data.avmnist import AVMNISTDataset
    from pavlov.models.lightning_module import PavlovLightningModule
    from pavlov.utils.checkpoint import load_model_from_checkpoint

    model = load_model_from_checkpoint(
        PavlovLightningModule,
        checkpoint_path,
        map_location="cpu",
    )
    model.eval()

    test_dir = Path(data_dir) / "avmnist" / "test"
    if not test_dir.exists():
        raise FileNotFoundError(
            f"Test data not found at {test_dir}. Run pavlov-download first."
        )
    dataset = AVMNISTDataset(test_dir)
    return model, dataset


def spectrogram_to_image(spec: torch.Tensor | np.ndarray) -> np.ndarray:
    """Render a (1,112,112) spectrogram as an RGB numpy image."""
    if isinstance(spec, torch.Tensor):
        spec = spec.detach().cpu().numpy()
    if spec.ndim == 3:
        spec = spec.squeeze(0)

    fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=72)
    ax.imshow(spec, aspect="auto", origin="lower", cmap="magma")
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.canvas.draw()

    # buffer_rgba() is the modern replacement for the removed tostring_rgb()
    w, h = fig.canvas.get_width_height()
    buf = np.asarray(fig.canvas.buffer_rgba()).reshape(h, w, 4)
    img = buf[:, :, :3].copy()  # drop alpha, copy to own memory
    plt.close(fig)
    return img


def vision_to_image(tensor: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert a (1,28,28) vision tensor to (28,28) uint8 numpy array."""
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    if tensor.ndim == 3:
        tensor = tensor.squeeze(0)
    return (tensor * 255).clip(0, 255).astype(np.uint8)


@torch.no_grad()
def run_inference(model, sample: dict) -> dict:
    """Run a single sample through the model, returning all reconstructions.

    Returns dict with keys like ("vision", "vision"), ("vision", "audio"), etc.
    mapping to reconstruction tensors.
    """
    # Batch dimension
    inputs = {
        "vision": sample["vision"].unsqueeze(0),
        "audio": sample["audio"].unsqueeze(0),
    }
    result = model.model.forward(inputs)
    recons = {}
    for src in ("vision", "audio"):
        for tgt in ("vision", "audio"):
            recons[(src, tgt)] = result["reconstructions"][src][tgt].squeeze(0)
    return recons


def build_app(model, dataset):
    """Build and return the Gradio app."""
    import gradio as gr

    from pavlov.evaluation.audio_utils import AVMNIST_SR, spectrogram_to_audio

    def process_vision():
        idx = random.randint(0, len(dataset) - 1)
        sample = dataset[idx]
        label = sample["label"].item()
        recons = run_inference(model, sample)

        orig_img = vision_to_image(sample["vision"])
        same_img = vision_to_image(recons[("vision", "vision")])
        cross_spec_img = spectrogram_to_image(recons[("vision", "audio")])
        cross_audio = spectrogram_to_audio(recons[("vision", "audio")])

        return (
            f"Digit: {label}",
            orig_img,
            same_img,
            cross_spec_img,
            (AVMNIST_SR, cross_audio),
        )

    def process_audio():
        idx = random.randint(0, len(dataset) - 1)
        sample = dataset[idx]
        label = sample["label"].item()
        recons = run_inference(model, sample)

        orig_spec_img = spectrogram_to_image(sample["audio"])
        orig_audio = spectrogram_to_audio(sample["audio"])
        same_spec_img = spectrogram_to_image(recons[("audio", "audio")])
        same_audio = spectrogram_to_audio(recons[("audio", "audio")])
        cross_img = vision_to_image(recons[("audio", "vision")])

        return (
            f"Digit: {label}",
            orig_spec_img,
            (AVMNIST_SR, orig_audio),
            same_spec_img,
            (AVMNIST_SR, same_audio),
            cross_img,
        )

    with gr.Blocks(title="Pavlov Demo") as app:
        gr.Markdown("# Pavlov Multimodal Reconstruction Demo")

        with gr.Tab("Vision Input"):
            v_btn = gr.Button("New Sample")
            v_label = gr.Textbox(label="Label", interactive=False)
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Original Image**")
                    v_orig = gr.Image(label="Original", type="numpy")
                with gr.Column():
                    gr.Markdown("**Vision → Vision**")
                    v_same = gr.Image(label="Same-modal", type="numpy")
                with gr.Column():
                    gr.Markdown("**Vision → Audio**")
                    v_cross_spec = gr.Image(label="Cross-modal spectrogram", type="numpy")
                    v_cross_audio = gr.Audio(label="Cross-modal audio")

            v_btn.click(
                fn=process_vision,
                outputs=[v_label, v_orig, v_same, v_cross_spec, v_cross_audio],
            )

        with gr.Tab("Audio Input"):
            a_btn = gr.Button("New Sample")
            a_label = gr.Textbox(label="Label", interactive=False)
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Original Spectrogram**")
                    a_orig_spec = gr.Image(label="Original", type="numpy")
                    a_orig_audio = gr.Audio(label="Original audio")
                with gr.Column():
                    gr.Markdown("**Audio → Audio**")
                    a_same_spec = gr.Image(label="Same-modal spectrogram", type="numpy")
                    a_same_audio = gr.Audio(label="Same-modal audio")
                with gr.Column():
                    gr.Markdown("**Audio → Vision**")
                    a_cross_img = gr.Image(label="Cross-modal image", type="numpy")

            a_btn.click(
                fn=process_audio,
                outputs=[a_label, a_orig_spec, a_orig_audio, a_same_spec, a_same_audio, a_cross_img],
            )

    return app


def main():
    parser = argparse.ArgumentParser(description="Pavlov interactive demo")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to model checkpoint (auto-detects best if omitted)",
    )
    parser.add_argument(
        "--data-dir", type=str, default="data",
        help="Path to data directory containing avmnist/",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to (e.g. 0.0.0.0)")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    args = parser.parse_args()

    ckpt = args.checkpoint or find_best_checkpoint()
    print(f"Loading model from {ckpt}...")
    model, dataset = load_demo_model(ckpt, args.data_dir)
    print(f"Loaded. Test set has {len(dataset)} samples.")

    app = build_app(model, dataset)
    app.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
