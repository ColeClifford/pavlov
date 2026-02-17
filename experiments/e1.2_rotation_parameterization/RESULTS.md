# E1.2 Results: Rotation Parameterization Comparison

*Completed: 2026-02-17*

## Summary

**Key Finding:** Simple transforms perform nearly as well as SO(d) matrix exponential, with dramatically lower computational cost.

| Variant | Val Loss | Train Loss | Params (d=128) | Runtime | Winner? |
|---------|----------|------------|----------------|---------|---------|
| **scaling_bias** | 1.709 | 1.522 | 256 | 750s | Simplest baseline |
| **householder** | 1.706 | 1.523 | 256 | 757s | 🥈 Tied for best |
| **linear** | 1.708 | 1.540 | 16,384 | 776s | High capacity, needs regularization |
| **block_rotation** | 1.706 | 1.523 | 64 | 752s | 🥇 Best efficiency |
| *SO(d) baseline* | ~1.71* | ~1.52* | 8,128 | ~800s* | E1.1 reference |

*\*E1.1 baseline values for comparison*

## Detailed Metrics

### Scaling + Bias (simplest)
```
transform_type: scaling_bias
Val Loss: 1.709 | Train Loss: 1.522
Cross-modal Recon: 0.0356
Same-modal Recon: 0.0150
Contrastive: 3.318
Scale mean: vision=0.75, audio=0.57
Bias mean: vision≈0, audio≈0
```

### Householder Reflections (orthogonal)
```
transform_type: householder
Val Loss: 1.706 | Train Loss: 1.523
Cross-modal Recon: 0.0356
Same-modal Recon: 0.0093
Contrastive: 3.322
```
- Maintains orthogonality via reflections
- k=2 reflections sufficient for good performance

### Unconstrained Linear
```
transform_type: linear
Val Loss: 1.708 | Train Loss: 1.540
Cross-modal Recon: 0.0345
Same-modal Recon: 0.0166
Contrastive: 3.313
Condition numbers: vision=995,768 | audio=3,086,307
```
- ⚠️ Very high condition numbers indicate near-singular matrices
- Orthogonality regularization helped but matrices still ill-conditioned
- Most parameters (d²) but not best performance

### Block-Diagonal 2D Rotations (RoPE-style) ⭐
```
transform_type: block_rotation
Val Loss: 1.706 | Train Loss: 1.523
Cross-modal Recon: 0.0359
Same-modal Recon: 0.0119
Contrastive: 3.317
Angle stats: vision_mean=3.26 | audio_mean=3.02
```
- **Winner on efficiency**: 64 params vs 8,128 for SO(d)
- Tied for best validation loss
- Angles learned to be ~π (modalities use opposite rotations)

## Analysis

### Does Orthogonality Matter?
**Partially.** Householder and block_rotation (both orthogonal) tied for best. But scaling_bias (non-orthogonal) was only 0.003 worse — practically negligible.

### Does Full SO(d) Flexibility Matter?
**No.** Block-diagonal 2D rotations (d/2 independent angles) match full SO(d) performance while being ~100x more parameter-efficient and O(d) vs O(d³) compute.

### Is Linear Transform Worth It?
**No.** Despite having d² parameters, linear transform:
- Didn't outperform simpler methods
- Developed numerical instability (high condition numbers)
- Required careful regularization

## Recommendations

1. **For production:** Use `block_rotation` — best efficiency/performance tradeoff
2. **For simplicity:** `scaling_bias` is nearly as good with trivial implementation
3. **Avoid:** `linear` unless you need maximum expressiveness and can handle instability
4. **SO(d) via matrix_exp:** No longer recommended — too expensive for equivalent results

## Implications for Pavlov

The E1.1 finding (rotations are critical) combined with E1.2 (simple rotations work) suggests:

> **The key is modality-specific conditioning, not the complexity of the transform.**

Block-diagonal rotations provide:
- Orthogonality (preserves distances in shared space)
- Per-dimension learned angles (captures modality structure)
- O(d) compute and O(d/2) parameters

This validates a much simpler architecture than originally proposed.

## Next Steps

1. **E1.3:** Loss component ablation — which losses actually matter?
2. **E4.1:** CLIP baseline comparison — how does Pavlov compare to contrastive-only?
3. **E5.1:** Scale to CREMA-D dataset — does this hold on harder tasks?

## WandB Runs

- scaling_bias: `run-20260217_021044-9i62727a`
- householder: `run-20260217_022325-7938sqrk`
- linear: `run-20260217_024121-i3oaj3tp`
- block_rotation: `run-20260217_025457-cn18cw7s`

---

*Analysis by Marvin + Prism | Data from WandB*
