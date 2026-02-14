"""Evaluation entry point for Pavlov."""

from __future__ import annotations

from pathlib import Path

import hydra
import pytorch_lightning as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

# Resolve config path relative to package root (works when run via entry point)
_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "configs"


@hydra.main(version_base=None, config_path=str(_CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed)

    from pavlov.data.avmnist import AVMNISTDataModule
    from pavlov.evaluation.cross_modal_transfer import evaluate_cross_modal_transfer
    from pavlov.evaluation.retrieval import evaluate_retrieval
    from pavlov.evaluation.sample_logging import log_samples_from_dataloader
    from pavlov.models.lightning_module import PavlovLightningModule
    from pavlov.utils.checkpoint import load_model_from_checkpoint

    # Load from checkpoint (handles torch.compile _orig_mod keys)
    if not hasattr(cfg, "checkpoint") or cfg.checkpoint is None:
        raise ValueError("Must specify +checkpoint=<path> to a model checkpoint.")

    model = load_model_from_checkpoint(
        PavlovLightningModule,
        cfg.checkpoint,
        weights_only=False,
    )
    model.eval()

    datamodule = AVMNISTDataModule(**cfg.data)
    datamodule.setup("test")
    test_loader = datamodule.test_dataloader()

    modalities = list(cfg.modalities)

    # Cross-modal transfer (run first for per-digit logging)
    transfer_results = {}
    for train_mod in modalities:
        for test_mod in modalities:
            if train_mod == test_mod:
                continue
            transfer_results[(train_mod, test_mod)] = evaluate_cross_modal_transfer(
                model, test_loader, train_mod, test_mod
            )

    # Cross-modal retrieval (run first for per-digit logging)
    retrieval_results = {}
    for query_mod in modalities:
        for gallery_mod in modalities:
            if query_mod == gallery_mod:
                continue
            retrieval_results[(query_mod, gallery_mod)] = evaluate_retrieval(
                model, test_loader, query_mod, gallery_mod, k=10
            )

    # Missing-modality (if enabled, run once for both print and TensorBoard)
    missing_mod_results = {}
    if cfg.get("eval_missing_modality", False):
        from pavlov.evaluation.missing_modality import evaluate_missing_modality

        for mod in modalities:
            missing_mod_results[mod] = evaluate_missing_modality(
                model, test_loader, mod, modalities
            )

    # Optional: log diagnostics to TensorBoard
    if cfg.get("log_tensorboard", False):
        import matplotlib.pyplot as plt

        from torch.utils.tensorboard import SummaryWriter

        from pavlov.evaluation.modality_alignment import log_modality_alignment
        from pavlov.evaluation.reconstruction_metrics import compute_reconstruction_mse
        from pavlov.evaluation.rotation_viz import log_rotation_matrices
        from pavlov.evaluation.visualization import log_tsne_embeddings

        log_dir = cfg.get("log_dir") or HydraConfig.get().runtime.output_dir
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        with SummaryWriter(log_dir=str(log_dir)) as writer:
            log_samples_from_dataloader(
                model, test_loader, writer, step=0, modalities=modalities
            )
            mse_metrics = compute_reconstruction_mse(model, test_loader, modalities)
            for key, val in mse_metrics.items():
                writer.add_scalar(f"eval/mse/{key}", val, 0)
            log_rotation_matrices(model, writer, 0, modalities)
            log_modality_alignment(model, test_loader, writer, 0, modalities)
            log_tsne_embeddings(model, test_loader, writer, 0, modalities, max_samples=500)
            # Per-digit transfer bar charts
            for (train_mod, test_mod), res in transfer_results.items():
                per_class = res.get("per_class_transfer_accuracy", {})
                if per_class:
                    fig, ax = plt.subplots(figsize=(10, 4))
                    digits = sorted(per_class.keys())
                    accs = [per_class[d] for d in digits]
                    ax.bar(digits, accs)
                    ax.set_xlabel("Digit")
                    ax.set_ylabel("Transfer Accuracy")
                    ax.set_title(f"Transfer {train_mod} -> {test_mod} (per digit)")
                    writer.add_figure(
                        f"eval/transfer_per_digit_{train_mod}_{test_mod}",
                        fig,
                        0,
                    )
                    plt.close(fig)
            # Per-digit retrieval bar charts
            for (query_mod, gallery_mod), res in retrieval_results.items():
                per_class = res.get("per_class_recall_at_k", {})
                if per_class:
                    fig, ax = plt.subplots(figsize=(10, 4))
                    digits = sorted(per_class.keys())
                    recalls = [per_class[d] for d in digits]
                    ax.bar(digits, recalls)
                    ax.set_xlabel("Digit")
                    ax.set_ylabel("Recall@10")
                    ax.set_title(f"Retrieval {query_mod} -> {gallery_mod} (per digit)")
                    writer.add_figure(
                        f"eval/retrieval_per_digit_{query_mod}_{gallery_mod}",
                        fig,
                        0,
                    )
                    plt.close(fig)
            # Missing-modality (if enabled)
            if missing_mod_results:
                for mod, res in missing_mod_results.items():
                    writer.add_scalar(
                        f"eval/missing_modality/{mod}/accuracy",
                        res["accuracy"],
                        0,
                    )
                    writer.add_scalar(
                        f"eval/missing_modality/{mod}/cross_recon_mse",
                        res["cross_recon_mse"],
                        0,
                    )
        print(f"Logged diagnostics to TensorBoard at {log_dir}")

    print("=" * 60)
    print("Pavlov Evaluation Results")
    print("=" * 60)

    for (train_mod, test_mod), results in transfer_results.items():
        print(f"\nCross-modal transfer: {train_mod} -> {test_mod}")
        print(f"  Train accuracy:    {results['train_accuracy']:.4f}")
        print(f"  Transfer accuracy: {results['transfer_accuracy']:.4f}")

    for (query_mod, gallery_mod), results in retrieval_results.items():
        print(f"\nRetrieval: {query_mod} -> {gallery_mod}")
        for key, val in results.items():
            if key not in ("per_class_recall_at_k", "per_class_transfer_accuracy"):
                print(f"  {key}: {val:.4f}")

    # Missing-modality robustness (optional)
    if missing_mod_results:
        print("\n" + "-" * 40)
        print("Missing-Modality Robustness")
        print("-" * 40)
        for mod, res in missing_mod_results.items():
            print(f"\n  Available: {mod} only")
            print(f"    Accuracy (linear probe): {res['accuracy']:.4f}")
            print(f"    Cross-recon MSE (fill-in): {res['cross_recon_mse']:.6f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
