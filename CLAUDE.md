# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Pavlov** is a research project exploring unified multimodal embeddings with geometric modality conditioning. The core idea: a single shared embedding matrix transformed by modality-specific learned rotation matrices (parameterized via Lie algebra so(d)), trained with joint and cross-modal reconstruction objectives. Named after Pavlovian conditioning as an analogy for cross-modal associative binding.

## Current State

The repository is in the **pre-implementation research phase**. It contains a literature review artifact covering:
- Shared multimodal representations (Meta-Transformer, VATT, Perceiver, data2vec)
- Rotation-based conditioning (RoPE, LieRE, ComRoPE, Circle-RoPE)
- Cross-modal reconstruction (multimodal VAEs: MVAE, MMVAE, MoPoE-VAE, DMVAE)
- Hebbian/associative learning connections to cross-modal binding

## Planned Architecture

- **Shared embedding matrix** W ∈ ℝ^(d×k)
- **Modality-specific rotations** R_v, R_a ∈ SO(k), parameterized via skew-symmetric matrices (Lie algebra so(k))
- **Encoding**: R_modality · W · tokenize(input) per modality
- **Training**: Joint reconstruction + cross-modal reconstruction losses, optional contrastive alignment and orthogonality regularization
- **Proof-of-concept dataset**: AV-MNIST (~70K paired MNIST images + spoken digit spectrograms from FSDD)

## Evaluation Protocol

- Cross-modal transfer (train on modality A, test on modality B)
- Cross-modal reconstruction quality (FID / MSE)
- t-SNE/UMAP visualization (embeddings colored by class and modality)
- Missing-modality robustness
- Cross-modal retrieval (Recall@K, MAP)

## Key References to Differentiate From

- **DMVAE** — private/shared latent decomposition
- **Circle-RoPE** — orthogonal modality subspaces in rotary embeddings
- **Modality gap hyperplane rotation** — theoretical proof that rotation alignment closes the modality gap
- **MultiBench** — standardized benchmarks and evaluation tooling for AV-MNIST
