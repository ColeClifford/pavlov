# Unified multimodal embeddings with geometric modality conditioning

**A single shared embedding transformed by modality-specific rotations is a largely unexplored but theoretically grounded idea at the intersection of three active research areas: unified multimodal architectures, geometric representation learning, and associative cross-modal binding.** The academic literature offers strong building blocks — from Meta-Transformer's frozen shared encoder processing 12 modalities, to Lie-algebraic rotary embeddings that condition attention on spatial structure, to multimodal VAEs that reconstruct across modalities from shared latent spaces — but no existing work directly combines a shared embedding matrix with learned rotation transforms for modality conditioning. This gap represents a genuine research opportunity. The standard proof-of-concept for this type of work is AV-MNIST: pairing handwritten digit images with spoken digit spectrograms, a setup with established baselines and open-source tooling.

---

## The landscape of truly shared multimodal representations

The field distinguishes between **joint representations** (single shared latent space) and **coordinated representations** (separate encoders aligned by loss functions like contrastive learning). CLIP, ImageBind, and similar contrastive models fall in the coordinated camp — they use separate encoders per modality and align outputs in a shared metric space. The user's idea falls squarely in the joint representation camp, where architectures share actual parameters across modalities.

Several architectures have pushed toward maximal parameter sharing. **Meta-Transformer** (Zhang et al., 2023) processes 12 modalities through a single frozen ViT encoder, using only lightweight modality-specific tokenizers to convert raw inputs into token sequences. **VATT** (Akbari et al., NeurIPS 2021) demonstrated a modality-agnostic transformer backbone shared across video, audio, and text, performing on par with separate modality-specific models. **Gato** (DeepMind, 2022) serializes data from 604 tasks across text, images, and continuous control into a single token stream processed by one decoder-only transformer with fully shared weights. **Unified-IO 2** (Allen AI, CVPR 2024) tokenizes images, text, audio, and actions into a shared discrete vocabulary, processing everything through a single encoder-decoder transformer.

The recurring architectural pattern is **modality-specific tokenization → shared transformer → task-specific heads**. The tokenizer converts raw signals into a common format (token sequences), and the shared transformer processes all modalities identically. A hybrid approach — shared self-attention with modality-specific feed-forward layers — has emerged as a practical sweet spot, exemplified by **BEiT-3** (Microsoft, CVPR 2023) and **ONE-PEACE** (Alibaba, 2023). DeepMind's **Perceiver** (ICML 2021) takes a different approach: a modality-agnostic architecture using cross-attention to map arbitrary inputs to a fixed-size latent bottleneck, then processing through shared self-attention, decoupling compute from input size entirely.

**data2vec** (Meta AI, ICML 2022) unifies the *learning objective* rather than the architecture: the same self-distillation method (predicting contextualized latent representations from masked views) achieves competitive results across speech, vision, and text, though it still uses modality-specific feature extractors.

---

## Rotation-based conditioning is emerging but not yet applied to modality transforms

The idea of using rotation matrices to condition shared representations on modality type connects to a rapidly expanding body of work on **rotary position embeddings (RoPE)** and their multimodal extensions. The foundational **RoFormer** (Su et al., 2021) applies block-diagonal 2×2 rotation matrices to encode absolute position while preserving relative position information in self-attention. **LieRE** (Ostmeier et al., ICML 2025) generalizes this to learned dense high-dimensional rotations via the Lie algebra so(d), parameterizing trainable rotations through exponentiation of skew-symmetric matrices. **ComRoPE** (Yu et al., CVPR 2025) addresses commutativity constraints, proving that rotation difference R(x)ᵀR(y) can represent location difference y−x if and only if angle matrices pairwise commute.

For multimodal applications, **Circle-RoPE** (Wang et al., 2025) projects different modalities into **orthogonal affine subspaces** before applying rotary encoding, explicitly preventing cross-axis interference between modalities. **VideoRoPE** (Wei et al., 2025) allocates different frequency bands to temporal versus spatial dimensions. These works establish the mathematical framework — Lie group SO(d) rotations applied to embedding spaces — but they condition on *position*, not on *modality identity*. The user's idea of rotating a shared embedding by modality-specific rotation matrices to produce modality-conditioned representations is a natural but unstudied extension.

Three additional lines of work provide theoretical grounding. First, research on the **modality gap** in CLIP (Liang et al., NeurIPS 2022) discovered that image and text embeddings occupy completely separate narrow cones on the hypersphere. A 2025 paper ("Decipher the Modality Gap") formally proves that **hyperplane rotation** — rotating one modality's embedding hyperplane to align with another — can achieve perfect cross-modal alignment. Second, **Canonical Correlation Analysis (CCA)** and its deep extensions learn modality-specific linear projections that maximize cross-modal correlation, operating in the inverse direction (modality → shared space rather than shared → modality). Third, neuroscience research (Flesch et al., *Neuron* 2022) shows that neural networks in the "rich learning regime" project task representations onto **low-dimensional orthogonal manifolds**, with context-dependent rotation of stimulus axes — remarkably close to the proposed approach.

**Hyperbolic geometry** offers another non-Euclidean alternative. **MERU** (Facebook Research, ICML 2023) places image-text representations on the Lorentz hyperboloid, naturally encoding visual-semantic hierarchy (text near origin, images farther out). While not rotation-based, MERU demonstrates that non-Euclidean geometric structure can improve multimodal representations.

---

## Cross-modal reconstruction from shared latent spaces has deep roots

The reconstruction-based approach to multimodal learning predates contrastive methods and provides the closest existing framework to the user's envisioned architecture. **Ngiam et al. (ICML 2011)** — "Multimodal Deep Learning" — is the foundational paper: bimodal deep autoencoders trained to reconstruct *both* modalities given only one as input. They achieved cross-modal transfer on audio-visual digit recognition (67% accuracy on CUAVE, training on audio, testing on video, versus 10% chance), replicated the McGurk effect, and demonstrated that the shared middle-layer representation captured modality-invariant features.

**Srivastava and Salakhutdinov (NeurIPS 2012)** extended this with multimodal Deep Boltzmann Machines, learning a joint generative model P(v_img, v_txt | θ) that could fill in missing modalities by sampling from conditional distributions. Their desiderata for multimodal representations remain canonical: similarity in representation space should reflect conceptual similarity, representations should be obtainable even with missing modalities, and it should be possible to fill in missing modalities from observed ones.

The multimodal VAE family offers the most direct technical framework:

- **MVAE** (Wu & Goodman, NeurIPS 2018) uses a **Product-of-Experts** to combine modality-specific posteriors into a joint posterior, enabling cross-modal generation through a shared latent space
- **MMVAE** (Shi et al., NeurIPS 2019) replaces PoE with **Mixture-of-Experts**, avoiding the "veto phenomenon" where one modality's low posterior suppresses the joint
- **MoPoE-VAE** (Sutter et al., ICLR 2021) generalizes both as **Mixture-of-Products-of-Experts**, providing a provably tighter ELBO bound
- **DMVAE** (Lee & Pavlovic, CVPR 2021 Workshop) explicitly **disentangles shared and private** latent factors, enabling cross-modal synthesis by transferring shared codes while sampling modality-specific codes from priors

A 2025 paper on reconstruction-driven multimodal autoencoders (MMAE) confirms that joint reconstruction losses across modalities remain a practical and data-efficient alternative to contrastive approaches, using a shared 128-dimensional bottleneck across image, audio, and text decoders.

---

## Pavlovian and Hebbian associations have computational parallels

The user's intuition about Pavlovian conditioning maps onto a rich literature connecting biological associative learning with computational cross-modal binding. **Vincis and Fontanini (eLife, 2016)** provide direct experimental evidence: recording gustatory cortex neurons before and after classical conditioning (cue-taste pairing), they found that **Pavlovian conditioning directly reshapes cross-modal neural representations** — previously unimodal neurons became cross-modally responsive, and representation overlap between modalities increased. Stanford research (Grewe, Schnitzer et al., *Nature* 2017) observed the same in amygdala: during fear conditioning, tone-responsive neurons began to resemble shock-responsive neurons, creating shared representations for associated stimuli.

Computationally, **Hebbian learning** ("fire together, wire together") serves as the bridge between biological associative learning and modern cross-modal objectives. **Cuppini et al. (2012)** modeled multisensory integration in the superior colliculus via Hebbian learning, reproducing the developmental trajectory from unimodal to cross-modal responsiveness. **Kaur et al. (2022)** built a practical cross-modal retrieval system using two Self-Organizing Maps (one per modality) linked by a Hebbian network — co-activated nodes across modality-specific maps form associative links enabling cross-modal retrieval without backpropagation.

The formal connection between these approaches runs through **energy-based models**. Multimodal Deep Boltzmann Machines (Srivastava & Salakhutdinov) use energy-based learning with contrastive divergence — formally related to Hebbian principles. **Modern Hopfield networks** (Ramsauer et al., 2020) proved equivalent to transformer attention mechanisms, bridging associative memory directly to contemporary architectures. Krotov and Hopfield's **Dense Associative Memory** (2016) extends this with higher-order interactions, dramatically increasing capacity. A 2019 ICANN paper explicitly extends Hopfield networks to store and retrieve multimodal patterns using Hebbian weight updates.

**Friston's predictive coding framework** offers the most comprehensive unification: cross-modal associations form when co-occurring sensory events minimize joint prediction error, with weight update rules that are local and Hebbian-like. The reconstruction objective in multimodal autoencoders is formally equivalent to prediction error minimization in this framework.

The proper academic terms for this intersection are: **cross-modal associative learning**, **Hebbian multimodal binding**, **cross-modal plasticity**, and **multisensory integration**. The Rescorla-Wagner learning rule (ΔV = αβ(λ − ΣV)) from classical conditioning is formally equivalent to the delta rule in neural networks, providing a direct mathematical bridge.

---

## AV-MNIST is the standard proof-of-concept starting point

For a digit-based multimodal experiment, the established benchmark is **AV-MNIST** from MultiBench (Liang et al., NeurIPS 2021). It pairs **MNIST handwritten digit images** (28×28 grayscale) with **spoken digit spectrograms** (112×112) from the Free Spoken Digit Dataset (FSDD), yielding ~70K paired samples across 10 digit classes. The standard protocol uses LeNet-3 for image encoding (48-dim output) and LeNet-5 for audio spectrogram encoding (192-dim output), with cross-entropy classification loss.

Key datasets for scaling beyond the initial proof-of-concept:

- **AudioMNIST**: 30,000 spoken digit recordings from 50 speakers, enabling speaker-invariant digit representation learning
- **N-MNIST + SHD**: Neuromorphic event-based versions for spiking neural network approaches (Bjorndahl et al., 2024 achieved 98.43% with late fusion)
- **CUAVE**: Audio-visual connected digit strings from 36 speakers, used in the foundational Ngiam et al. work
- **AVLetters**: Audio-visual letters A-Z, for extending to alphabet domains

The standard evaluation protocol for validating that a representation is truly multimodal includes five tests. **Cross-modal transfer** (train classifier on modality A, test on modality B) is the gold standard — Ngiam et al.'s 67% audio→video transfer versus 10% chance remains the classic demonstration. **Cross-modal retrieval** (Recall@K, MAP) measures whether embeddings from different modalities cluster by semantic content. **Missing modality robustness** evaluates graceful degradation. **Cross-modal reconstruction** (generate one modality from another through shared representation) directly tests the generative quality. **t-SNE/UMAP visualization** provides qualitative confirmation that same-class embeddings cluster regardless of source modality. MultiBench provides standardized implementations for all of these.

---

## Recommended experimental design and literature search terms

For the proposed experiment — a shared embedding matrix with modality-specific rotation transforms, trained with reconstruction objectives on digit data — the following design synthesizes best practices from the literature:

**Architecture**: A shared embedding matrix W ∈ ℝ^(d×k) with two learned rotation matrices R_v, R_a ∈ SO(k) (parameterized via skew-symmetric matrices in the Lie algebra so(k), following LieRE's approach). Visual input encoded as R_v · W · tokenize(image); audio input encoded as R_a · W · tokenize(spectrogram). Modality-specific lightweight tokenizers convert raw inputs to a common dimensionality. The shared embedding captures cross-modal semantic content; the rotations capture modality-specific structure.

**Training objective**: Joint reconstruction loss (reconstruct both modalities from shared representation, following Ngiam et al.) plus cross-modal reconstruction (reconstruct audio from visually-derived embedding and vice versa, following DMVAE). Optionally add contrastive alignment loss to encourage same-digit embeddings to cluster. An orthogonality regularizer on R_v and R_a (following OSF and DMHOR) could encourage complementary modality-specific transforms.

**Evaluation**: Cross-modal transfer accuracy, cross-modal reconstruction quality (FID or pixel-level MSE), t-SNE visualization of shared embeddings colored by digit class and modality, and missing-modality robustness.

For literature searches, the most productive query terms are:

- **"joint multimodal representation"** + "shared latent space" (finds autoencoder/VAE work)
- **"modality-agnostic encoder"** or "unified multimodal transformer" (finds Meta-Transformer, VATT, Perceiver)
- **"rotary position embedding multimodal"** or "Lie group rotation embedding" (finds LieRE, ComRoPE, Circle-RoPE)
- **"cross-modal reconstruction"** + autoencoder (finds Ngiam et al., MMAE line of work)
- **"multimodal VAE"** or "MVAE" or "MMVAE" (finds the VAE family)
- **"Hebbian cross-modal"** or "associative multimodal learning" (finds neuroscience-inspired work)
- **"factorized multimodal representation"** (finds MFM, FactorCL — shared vs. modality-specific factor decomposition)
- **"modality gap"** + rotation or alignment (finds hyperplane rotation alignment work)

## Conclusion

The proposed architecture — shared embedding + modality-specific rotations + reconstruction objectives — sits at a genuine gap in the literature. Existing work has independently established that shared transformer weights generalize across 12+ modalities (Meta-Transformer), that Lie-algebraic rotation matrices effectively condition embeddings on structural properties (LieRE/ComRoPE), that cross-modal reconstruction from shared latent spaces enables powerful associative transfer (multimodal VAEs and autoencoders), and that biological cross-modal binding follows Hebbian/associative principles formalized by the Rescorla-Wagner rule. No published work combines these elements into a single framework where a shared embedding matrix is rotated by learned modality-specific transforms and trained with joint reconstruction. The AV-MNIST benchmark provides the ideal starting point, with **~70K paired samples, established baselines for comparison, and standardized evaluation protocols** through MultiBench. The closest related architectures to cite and differentiate from are DMVAE (for private-shared decomposition), Circle-RoPE (for orthogonal modality subspaces in rotary embeddings), and the modality gap hyperplane rotation proof (for theoretical grounding that rotation-based alignment is sufficient for closing the modality gap).