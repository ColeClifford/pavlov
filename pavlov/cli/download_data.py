"""CLI entry point for downloading and preparing the AV-MNIST dataset."""

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from pavlov.data.download import download_and_prepare

log = logging.getLogger(__name__)

# Resolve config path relative to package root (works when run via entry point)
_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "configs"


@hydra.main(version_base=None, config_path=str(_CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    """Download and prepare the AV-MNIST dataset."""
    log.info("Starting AV-MNIST download and preparation ...")
    download_and_prepare(cfg.data.data_dir)
    log.info("Done.")


if __name__ == "__main__":
    main()
