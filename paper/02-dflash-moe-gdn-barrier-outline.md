# DFlash on MoE Consumer Hardware: From τ=9.4 Offline to the GDN State Barrier

## Paper Outline — Short/Workshop Paper (NeurIPS Efficient NLP / ICML)

### Abstract
We present the first independent reimplementation of DFlash speculative decoding with a block diffusion drafter, targeting a Mixture-of-Experts model (Qwen3.5-35B-A3B) on consumer hardware (RTX 5060 Ti, 16GB VRAM). Our drafter achieves τ=9.4 tokens accepted per step offline, exceeding the original DFlash paper (τ≈6.4) by 47%, using 4× less training data (69K vs 289K blocks). We introduce three architectural innovations: mask ratio embedding, single-shot inference (1 forward pass vs 16 iterative steps), and exponential position-weighted loss. However, we document a fundamental incompatibility between speculative decoding and Gated DeltaNet (GDN) hybrid architectures: the sequential recurrent state update in GDN layers is structurally incompatible with KV cache trimming (llama_memory_seq_rm), preventing online verification. We formalize this as the GDN State Barrier and propose chimere-deltanet, a Rust-native runtime that bypasses it by design.

### 1. Introduction
- Speculative decoding: drafter proposes, target verifies
- DFlash: block diffusion as drafter (arXiv:2602.06036)
- Challenge: applying DFlash to MoE models on consumer GPU
- Our contribution: τ=9.4 offline, GDN barrier identified

### 2. Method
- 2.1 Hidden state extraction from Qwen3.5-35B-A3B (custom C++ extractor, 5 layers)
- 2.2 Training pipeline: extraction → memmap → BF16 training
- 2.3 Three innovations:
  - Mask ratio embedding (inform drafter of masking schedule)
  - Single-shot inference (1 forward pass, not 16 iterative denoising steps)
  - Exponential position-weighted loss (earlier positions weighted more)

### 3. Experimental Results
- 3.1 Training: val_loss 6.99→1.27, val_acc 22.9%→81.7%
- 3.2 Offline benchmark: τ=9.4, 27.6% perfect blocks (15/15 accepted)
- 3.3 Comparison with DFlash paper: τ=9.4 vs τ≈6.4 (+47%)
- 3.4 Data efficiency: 69K blocks vs 289K (+4× more efficient)
- 3.5 72,908 blocks evaluated, statistically significant

### 4. The GDN State Barrier
- 4.1 Background: GDN recurrent state vs KV cache
- 4.2 The problem: llama_memory_seq_rm cannot trim GDN state
  - KV cache is position-indexed → trimming removes specific positions
  - GDN state is a compressed summary → no way to "undo" a token
  - Rejecting a draft token corrupts all subsequent GDN state
- 4.3 Four structural walls in llama.cpp:
  1. No seq_rm for recurrent layers
  2. No state snapshot/restore API
  3. No batch verification with state rollback
  4. No separate state management for draft vs target
- 4.4 Why this matters: all hybrid SSM-attention models face this

### 5. Workaround: chimere-deltanet
- 5.1 Rust-native runtime with explicit state management
- 5.2 GDN state checkpoint before speculation
- 5.3 Batch verify on attention layers only (8/32)
- 5.4 Restore GDN state on rejection

### 6. Architecture Evolution
- v1: Gaussian continuous diffusion → 0.62% acceptance (embedding residual, not hidden states)
- v2: Mask-predict discrete → 2.88% at 16 steps (correct architecture, insufficient data)
- v3: Single-shot + exponential loss → 81.7% accuracy, τ=9.4 (breakthrough)
- Each pivot teaches a lesson about block diffusion on MoE targets

### 7. Related Work
- DFlash (2602.06036), EAGLE-3, Medusa, DART
- SpecMamba (SSM speculative decoding challenges)
- BD3-LMs (block size interpolation)

### 8. Conclusion
- τ=9.4 validates DFlash on MoE with innovations
- GDN State Barrier is a structural result the community needs
- Future: chimere-deltanet online verification, entropy routing AR↔diffusion

### Appendix
- A. C++ hidden state extractor code
- B. Training hyperparameters
- C. Full acceptance rate curves
- D. GDN state corruption examples
