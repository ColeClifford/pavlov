# Pavlov

Unified multimodal embeddings with geometric modality conditioning.

A single shared embedding matrix is transformed by modality-specific learned rotation matrices (parameterized via the Lie algebra so(d)). Cross-modal reconstruction forces the shared space to capture modality-invariant structure, inspired by Pavlovian associative conditioning.

## Architecture

```
Encode:  Input_m ──> Encoder_m(x) ──> W ──> R_m^T ──> z_m (latent)
Decode:  z ──> W_dec ──> Decoder_m ──> Reconstruction

Same-modal:    z_v ──> W_dec ──> Decoder_v ──> Recon_v
Cross-modal:   z_v ──> W_dec ──> Decoder_a ──> CrossRecon_a  (vision → audio)
```

- **W**: Shared encode projection (embed_dim -> latent_dim), learned jointly across modalities
- **W_dec**: Shared decode projection (latent_dim -> embed_dim), separate learned layer
- **R_v, R_a**: Modality-specific rotation matrices in SO(k), parameterized via skew-symmetric matrices in the Lie algebra so(k) and mapped to SO(k) via matrix exponential
- **Training objective**: Same-modal reconstruction + cross-modal reconstruction + optional contrastive alignment + optional rotation orthogonality regularization

## Installation

```bash
git clone https://github.com/<your-org>/pavlov.git
cd pavlov
pip install -e ".[dev]"

# Optional: W&B logging support
pip install -e ".[wandb]"
```

## Quick Start

### 1. Download AV-MNIST data

```bash
pavlov-download
```

### 2. Train

```bash
# Default config (TensorBoard logging)
pavlov-train

# With config overrides
pavlov-train training.max_epochs=50 model.latent_dim=64

# With W&B logging
pavlov-train training.logger=wandb training.wandb.project=my-project

# With experiment preset
pavlov-train +experiment=baseline
```

### 3. Evaluate

```bash
pavlov-eval +checkpoint=path/to/checkpoint.ckpt

# Full diagnostic suite: samples, audio, rotation matrices, t-SNE, per-digit metrics
pavlov-eval "+checkpoint=path/to/checkpoint.ckpt" +log_tensorboard=true

# Include missing-modality robustness evaluation
pavlov-eval "+checkpoint=path/to/checkpoint.ckpt" +log_tensorboard=true +eval_missing_modality=true
```

### 4. Visualize embeddings

```bash
pavlov-viz +checkpoint=path/to/checkpoint.ckpt +save_path=embeddings.png

# Also log sample reconstructions to TensorBoard
pavlov-viz "+checkpoint=path/to/checkpoint.ckpt" +save_path=embeddings.png +log_tensorboard=true
```

### 5. View TensorBoard

Training outputs (logs, checkpoints, Hydra config) are written to `outputs/YYYY-MM-DD/HH-MM-SS/` per run. During validation, the following are logged: sample reconstructions (images + audio), rotation matrix heatmaps, modality alignment metrics, and t-SNE (every 5 epochs). With `pavlov-eval +log_tensorboard=true`, the full diagnostic suite is logged. To view TensorBoard:

```bash
tensorboard --logdir outputs/
```

## Configuration

Pavlov uses [Hydra](https://hydra.cc/) for configuration management. All configs live in `configs/`.

```
configs/
  config.yaml            # Top-level defaults
  model/default.yaml     # Model architecture
  data/avmnist.yaml      # Dataset settings
  training/default.yaml  # Training hyperparameters
  experiment/baseline.yaml  # Experiment presets
```

### Override examples

```bash
# Change learning rate and batch size
pavlov-train training.lr=5e-4 data.batch_size=128

# Change embedding dimensions
pavlov-train model.embed_dim=512 model.latent_dim=256

# Enable contrastive loss
pavlov-train training.loss_weights.contrastive=0.1

# Disable orthogonality regularization
pavlov-train training.loss_weights.orthogonality=0.0
```

## Project Structure

```
pavlov/
  models/
    pavlov_model.py      # Core model: shared embeddings + rotation conditioning
    lightning_module.py   # PyTorch Lightning training wrapper
    encoders.py          # Modality-specific encoders
    decoders.py          # Modality-specific decoders
  data/
    avmnist.py           # AV-MNIST data module (MNIST images + spoken digits)
  utils/
    rotation.py          # Lie algebra so(d) -> SO(d) rotation utilities
  losses.py              # Reconstruction, contrastive, and orthogonality losses
  evaluation/
    cross_modal_transfer.py  # Train on modality A, test on modality B
    retrieval.py             # Cross-modal retrieval (Recall@K, MAP)
    visualization.py         # t-SNE embedding plots
  cli/
    train.py             # Training entry point
    eval.py              # Evaluation entry point
    viz.py               # Visualization entry point
    download_data.py     # Data download script
configs/
  ...                    # Hydra configuration files
```

## Evaluation Protocol

- **Cross-modal transfer**: Train a linear classifier on vision embeddings, test on audio embeddings (and vice versa). High transfer accuracy indicates modality-invariant representations.
- **Cross-modal retrieval**: Given a query from one modality, retrieve nearest neighbors in the other modality. Measured by Recall@K and Mean Average Precision.
- **t-SNE visualization**: Embeddings colored by digit class with different markers per modality. Well-aligned modalities show overlapping clusters.
- **Missing-modality robustness**: Evaluate performance when one modality is absent at test time.

## Adding a New Modality

1. **Create an encoder** in `pavlov/models/encoders.py`:
   ```python
   class MyEncoder(ModalityEncoder):
       def __init__(self, input_dim, hidden_dims, embed_dim):
           super().__init__()
           # Build your encoder layers

       def forward(self, x):
           # Return (batch, embed_dim) tensor
           return self.net(x)
   ```

2. **Create a decoder** in `pavlov/models/decoders.py`:
   ```python
   class MyDecoder(ModalityDecoder):
       def __init__(self, embed_dim, hidden_dims, output_dim):
           super().__init__()
           # Build your decoder layers

       def forward(self, z):
           # Return reconstructed input
           return self.net(z)
   ```

3. **Register the encoder/decoder** in the `build_encoder` and `build_decoder` factory functions.

4. **Add config** in `configs/model/default.yaml`:
   ```yaml
   my_modality_encoder:
     hidden_dims: [64, 128]
   my_modality_decoder:
     hidden_dims: [128, 64]
   ```

5. **Update modalities** list in `configs/config.yaml`:
   ```yaml
   modalities: [vision, audio, my_modality]
   ```

6. **Update your DataModule** to include the new modality in the batch dict.

## License

MIT
