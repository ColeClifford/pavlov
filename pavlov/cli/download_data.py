"""CLI entry point for downloading and preparing datasets.

Supports AV-MNIST (default) and CREMA-D. The dataset to download is
determined by ``cfg.data.dataset`` (set via Hydra config, e.g.
``pavlov-download data=cremad``).
"""

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

log = logging.getLogger(__name__)

# Resolve config path relative to package root (works when run via entry point)
_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "configs"


@hydra.main(version_base=None, config_path=str(_CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    """Download and prepare the selected dataset."""
    dataset = cfg.data.get("dataset", "avmnist")
    data_dir = cfg.data.data_dir

    if dataset == "avmnist":
        from pavlov.data.download import download_and_prepare

        log.info("Starting AV-MNIST download and preparation ...")
        download_and_prepare(data_dir)
    elif dataset == "cremad":
        from pavlov.data.cremad import prepare_cremad

        vision_size = cfg.data.get("vision_size", 64)
        log.info("Starting CREMA-D download and preparation ...")
        prepare_cremad(data_dir, vision_size=vision_size)
    else:
        raise ValueError(
            f"Unknown dataset: {dataset!r}. Available: 'avmnist', 'cremad'"
        )

    log.info("Done.")


if __name__ == "__main__":
    main()
