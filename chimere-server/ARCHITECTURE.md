# Chimère DeltaNet — Architecture

## Overview

chimere-deltanet is the Rust implementation of the Chimère inference engine.
It implements 5 core modules that together form a complete forward pass
for a hybrid linear-attention language model with entropy-adaptive compute.

**Status**: Executable specification. 53 tests passing, CPU-only.
Not yet connected to real model weights or CUDA.

```
chimere-deltanet/
├── src/
│   ├── lib.rs               # GatedDeltaNet — linear attention with delta rule
│   ├── moe_router.rs        # MoE — entropy-adaptive expert routing
│   ├── engram.rs             # Engram — hierarchical memory codebook
│   ├── hybrid_attention.rs   # Hybrid — DeltaNet + GQA blend with dynamic routing
│   └── block_diffusion.rs    # Block diffusion — masked diffusion scheduler
├── Cargo.toml
└── ARCHITECTURE.md           # This file
```

---

## Data Flow

A single forward pass through one Chimère layer:

```
Input: x [batch, seq_len, hidden_dim]
       state [batch, num_heads, head_dim, head_dim]  (per-layer, persistent)
       engram: Engram  (shared across layers)
│
├─1─► GatedDeltaNetLayer.forward(x, state)
│     ├── Pre-norm → Q,K,V projections → ShortConv1d → SiLU
│     ├── Gates: α(decay), β(update), g(output) via sigmoid
│     ├── Delta rule loop (recurrent, per timestep):
│     │     S' = α·S + β·(v - S·k)⊗k     ← associative memory update
│     │     o  = g · (S' · q)              ← gated recall
│     ├── Output projection + residual
│     └── Returns: (output_linear, updated_state)
│              [batch, seq_len, hidden_dim]
│              [batch, num_heads, head_dim, head_dim]
│
├─2─► compute_state_metrics(state)  →  StateMetrics per head
│     │   frobenius_norm: f32   — memory "fullness"
│     │   mean_delta: f32       — prediction error (set during forward)
│     │   effective_rank: f32   — memory utilization
│     │
│     ├──► Feeds into MoeRouter.adaptive_k()
│     ├──► Feeds into HybridAttentionLayer.compute_routing_scores()
│     └──► Feeds into Engram.maybe_update() (MDL gate)
│
├─3─► HybridAttentionLayer.forward(x, output_linear, state_metrics)
│     ├── compute_routing_scores(x, state_metrics)
│     │     Learned projection + mean_delta → sigmoid → [0,1] per token
│     ├── route_attention(scores, config)
│     │     Dynamic: top tokens above threshold, capped at capacity
│     ├── [if needed] GroupedQueryAttention.forward(x, active_mask)
│     │     GQA with token culling (inactive tokens masked to -inf)
│     ├── Blend: output = (1-w) * output_linear + w * output_gqa
│     └── Returns: (output_blended, AttentionRoutingDecision)
│              [batch, seq_len, hidden_dim]
│
├─4─► MoeRouter.route_token(output_blended, state_metrics)
│     ├── Gate projection → softmax with temperature
│     ├── tsallis_entropy(gate_probs, q=2) → routing_entropy
│     ├── adaptive_k(entropy, state_metrics)
│     │     Low entropy + low delta → K=1 (cheap)
│     │     High entropy + high delta → K=max_k (thorough)
│     ├── topk_select(probs, K) → expert_ids + weights
│     └── Returns: RoutingDecision
│           expert_ids: Vec<usize>, expert_weights: Vec<f32>
│
├─5─► Expert FFNs (not yet implemented)
│     ├── Shared expert (always active)
│     └── Routed experts (selected by MoeRouter)
│
├─6─► Engram.maybe_update(position, value, state_metrics)
│     ├── MDL gate: skip if mean_delta < threshold
│     ├── Deduplication: skip if poincare_distance < min_distance
│     ├── LRU eviction if at capacity
│     └── Returns: bool (was entry added?)
│
└─7─► [PLANNED] BlockDiffusionScheduler
      ├── Manages denoising over token blocks
      ├── Reads StateMetrics to adapt block boundaries
      └── Calls the full layer stack as its score model
```

---

## Module Interfaces

### 1. GatedDeltaNet (`lib.rs`)

The core linear attention mechanism. Each head maintains a persistent
state matrix S that acts as an associative memory.

#### Public Types

```rust
// Configuration
pub struct GatedDeltaNetConfig {
    pub hidden_dim: usize,     // 4096 (chimere/qwen3)
    pub num_heads: usize,      // 32
    pub head_dim: usize,       // 128
    pub conv_kernel: usize,    // 4
    pub gate_mode: GateMode,   // Scalar | ChannelWise
}

pub enum GateMode { Scalar, ChannelWise }

// State inspection — the central interface that connects all modules
pub struct StateMetrics {
    pub frobenius_norm: f32,   // ||S||_F — memory fullness
    pub mean_delta: f32,       // mean |v - S·k| — prediction error
    pub effective_rank: f32,   // tr(S^T S) / ||S||_F^2
}
```

#### Public Functions

```rust
// Layer (requires VarBuilder for weight initialization)
impl GatedDeltaNetLayer {
    pub fn new(config: GatedDeltaNetConfig, vb: VarBuilder) -> Result<Self>;
    pub fn forward(&self, x: &Tensor, state: Option<&Tensor>)
        -> Result<(Tensor, Tensor)>;
    // x:     [batch, seq_len, hidden_dim]
    // state: [batch, num_heads, head_dim, head_dim] or None (zeros)
    // Returns: (output, updated_state) — same shapes
}

// Standalone (no weights, for testing)
pub fn delta_rule_step(state: &Tensor, key: &Tensor, value: &Tensor,
                       alpha: f64, beta: f64) -> Result<(Tensor, Tensor)>;
// state: [d_v, d_k], key: [d_k], value: [d_v]
// Returns: (new_state [d_v, d_k], delta [d_v])

pub fn delta_rule_query(state: &Tensor, query: &Tensor) -> Result<Tensor>;
// state: [d_v, d_k], query: [d_k] → recalled [d_v]

pub fn compute_state_metrics(state: &Tensor) -> Result<StateMetrics>;
// state: [hd, hd] (single head)
```

#### Key Invariants

- Keys are L2-normalized before state updates (stable learning)
- Gates α, β ∈ (0,1) via sigmoid; g ∈ (0,1)^d via sigmoid
- State shape is always `[batch, num_heads, head_dim, head_dim]`
- Forward pass is **sequential** over time steps (recurrent delta rule)

---

### 2. MoE Router (`moe_router.rs`)

Entropy-adaptive expert selection. Unlike Qwen3's fixed top-2, the
number of active experts adapts per-token based on routing entropy
and DeltaNet state confidence.

#### Public Types

```rust
pub struct MoeRouterConfig {
    pub hidden_dim: usize,          // 4096
    pub num_experts: usize,         // 8 (routed)
    pub min_k: usize,               // 1
    pub max_k: usize,               // 4
    pub use_shared_expert: bool,    // true
    pub tsallis_q: f32,             // 2.0
    pub temperature: f32,           // 1.0
    pub routed_scaling_factor: f32, // 2.5
    pub sinkhorn_iterations: usize, // 20
    pub entropy_threshold_low: f32, // 0.3
    pub entropy_threshold_high: f32,// 0.7
}

pub struct RoutingDecision {
    pub expert_ids: Vec<usize>,       // selected expert indices
    pub expert_weights: Vec<f32>,     // normalized weights (sum=1)
    pub gate_probs: Vec<f32>,         // raw gate probabilities
    pub routing_entropy: f32,         // Tsallis entropy [0, 1]
    pub active_k: usize,             // how many experts chosen
}

pub struct BatchRoutingResult {
    pub decisions: Vec<RoutingDecision>,
    pub expert_load: Vec<f32>,        // fraction per expert
    pub mean_entropy: f32,
    pub mean_k: f32,
}
```

#### Public Functions

```rust
// Standalone (no weights)
pub fn tsallis_entropy(probs: &[f32], q: f32) -> f32;
    // Normalized to [0, 1]. q=2 standard, q→1 recovers Shannon.

pub fn shannon_entropy(probs: &[f32]) -> f32;

pub fn adaptive_k(routing_entropy: f32, state_metrics: Option<&StateMetrics>,
                  config: &MoeRouterConfig) -> usize;
    // Core link: StateMetrics.mean_delta modulates K
    // Low delta (confident) → reduces K by up to 50%

pub fn softmax_gate(logits: &[f32], temperature: f32) -> Vec<f32>;
pub fn topk_select(probs: &[f32], k: usize) -> (Vec<usize>, Vec<f32>);
pub fn sinkhorn_balance(logits: &[Vec<f32>], iterations: usize) -> Vec<Vec<f32>>;

// Layer (requires VarBuilder)
impl MoeRouter {
    pub fn new(config: MoeRouterConfig, vb: VarBuilder) -> Result<Self>;

    pub fn route_token(&self, hidden: &Tensor,
                       state_metrics: Option<&StateMetrics>)
        -> Result<RoutingDecision>;
    // hidden: [hidden_dim]

    pub fn route_batch(&self, hidden_states: &Tensor,
                       state_metrics: Option<&[StateMetrics]>)
        -> Result<BatchRoutingResult>;
    // hidden_states: [n_tokens, hidden_dim]
    // Uses Sinkhorn OT for load balancing
}
```

#### Key Invariants

- Tsallis entropy is normalized to [0, 1] (by max entropy for uniform)
- Gate probabilities sum to 1.0 (softmax)
- Top-K weights are renormalized after selection
- Sinkhorn produces doubly-stochastic assignment matrix
- `adaptive_k` clamps to `[min_k, max_k]`

---

### 3. Engram (`engram.rs`)

Shared hierarchical memory codebook addressed in the Poincaré ball.
MDL-gated: only stores new information when DeltaNet can't predict it.

#### Public Types

```rust
pub struct EngramConfig {
    pub entry_dim: usize,          // 128 (matches head_dim)
    pub max_entries: usize,        // 4096
    pub mdl_threshold: f32,        // 0.1
    pub max_poincare_norm: f32,    // 0.999
    pub top_k: usize,              // 4
    pub min_distance: f32,         // 0.5 (dedup threshold)
}

pub struct EngramEntry {
    pub position: Vec<f32>,        // Poincaré ball coordinates
    pub value: Vec<f32>,           // associated content vector
    pub access_count: u32,         // for LRU eviction
    pub mean_delta_at_write: f32,  // delta when entry was created
}

pub struct EngramQueryResult {
    pub entry_indices: Vec<usize>,
    pub distances: Vec<f32>,       // Poincaré distances
    pub values: Vec<Vec<f32>>,
    pub utilization: f32,          // entries / max_entries
}

pub struct EngramMetrics {
    pub utilization: f32,
    pub mean_depth: f32,           // mean Poincaré norm (0=general, 1=specific)
    pub mean_access_count: f32,
    pub recent_additions: usize,   // entries with access_count=0
}
```

#### Public Functions

```rust
// Poincaré ball operations (standalone)
pub fn poincare_distance(x: &[f32], y: &[f32]) -> f32;
    // d = arcosh(1 + 2||x-y||² / ((1-||x||²)(1-||y||²)))
pub fn poincare_project(x: &mut [f32], max_norm: f32);
pub fn poincare_depth(x: &[f32]) -> f32;  // ||x|| — hierarchy level
pub fn mobius_add(x: &[f32], y: &[f32]) -> Vec<f32>;
    // Proper hyperbolic translation, result projected into ball

// Codebook
impl Engram {
    pub fn new(config: EngramConfig) -> Self;

    pub fn query(&mut self, query_pos: &[f32]) -> EngramQueryResult;
    // query_pos: [entry_dim]
    // Returns top-k nearest entries, bumps their access_count

    pub fn maybe_update(&mut self, position: &[f32], value: &[f32],
                        state_metrics: &StateMetrics) -> bool;
    // MDL gate: skips if mean_delta < threshold
    // Dedup: skips if nearby entry exists (poincare_distance < min_distance)
    // Eviction: removes least-accessed entry if at capacity

    pub fn metrics(&self) -> EngramMetrics;
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
}
```

#### Key Invariants

- All positions have `||pos|| < max_poincare_norm` (projected on write)
- `query()` mutates access_count (for LRU tracking)
- MDL gate: `mean_delta >= mdl_threshold` required to store
- Deduplication: `poincare_distance >= min_distance` required
- Eviction: `swap_remove` on lowest `access_count`

---

### 4. Hybrid Attention (`hybrid_attention.rs`)

Combines GatedDeltaNet (O(n) linear) with GQA (O(n²) quadratic) in
a single layer. Dynamic routing decides per-token which mechanism(s)
to use, based on DeltaNet prediction error.

#### Public Types

```rust
pub enum RoutingMode {
    LinearOnly,   // Pure DeltaNet (28/32 layers in Chimère)
    FullOnly,     // Pure GQA (for ablation)
    Dynamic,      // Per-token routing based on StateMetrics
    FixedBlend,   // Both always, 50/50 (for ablation)
}

pub struct HybridAttentionConfig {
    pub hidden_dim: usize,           // 4096
    pub num_heads: usize,            // 32 (query heads)
    pub num_kv_heads: usize,         // 8 (KV heads, 4 groups)
    pub head_dim: usize,             // 128
    pub routing_mode: RoutingMode,
    pub delta_threshold: f32,        // 0.5 — above this → full attention
    pub full_attention_capacity: f32,// 0.25 — at most 25% tokens get GQA
}

pub struct AttentionRoutingDecision {
    pub full_attention_indices: Vec<usize>,  // which tokens got GQA
    pub full_attention_weights: Vec<f32>,    // per-token blend weight [0,1]
    pub full_attention_fraction: f32,        // fraction that used GQA
}

pub struct HybridStackConfig {
    pub num_layers: usize,           // 32
    pub attention_layers: Vec<usize>,// [7, 15, 23, 31] — 1:7 ratio
    pub dynamic_routing: bool,       // true for Chimère, false for Qwen3
}
```

#### Public Functions

```rust
// Standalone routing (no weights)
pub fn route_attention(delta_scores: &[f32], config: &HybridAttentionConfig)
    -> AttentionRoutingDecision;
    // Dynamic mode: scores above threshold, capped at capacity
    // Weight: sigmoid(5 * (score - threshold))

// GQA (requires VarBuilder)
impl GroupedQueryAttention {
    pub fn new(config: HybridAttentionConfig, vb: VarBuilder) -> Result<Self>;
    pub fn forward(&self, x: &Tensor, active_mask: Option<&[bool]>)
        -> Result<Tensor>;
    // x: [batch, seq_len, hidden_dim]
    // active_mask: [seq_len] — inactive tokens masked to -inf
    // Returns: [batch, seq_len, hidden_dim]
}

// Hybrid layer (requires VarBuilder)
impl HybridAttentionLayer {
    pub fn new(config: HybridAttentionConfig, vb: VarBuilder) -> Result<Self>;

    pub fn compute_routing_scores(&self, hidden: &Tensor,
                                   state_metrics: Option<&[StateMetrics]>)
        -> Result<Vec<f32>>;
    // Learned projection + mean_delta augmentation → sigmoid

    pub fn forward(&self, x: &Tensor, deltanet_output: &Tensor,
                   state_metrics: Option<&[StateMetrics]>)
        -> Result<(Tensor, AttentionRoutingDecision)>;
    // Runs GQA only on selected tokens, blends with deltanet_output
}

// Stack config
impl HybridStackConfig {
    pub fn chimere_32() -> Self;   // 4 attention layers at 7,15,23,31
    pub fn qwen3_48() -> Self;     // 12 attention layers, every 4th
    pub fn test_8() -> Self;       // 2 attention layers at 3,7
    pub fn routing_mode_for_layer(&self, layer_idx: usize) -> RoutingMode;
    pub fn attention_ratio(&self) -> f32;
}
```

#### Key Invariants

- GQA uses pre-norm (LayerNorm before projections)
- KV heads expanded via repeat (not concat) for GQA groups
- Causal mask: `mask[i][j] = 0 if j≤i, else -inf`
- Token culling mask: inactive keys get `-inf` (nobody attends to them)
- Capacity cap: at most `ceil(n * capacity)` tokens get GQA
- `FullOnly` mode still adds residual: `output = gqa(x) + x`
- `Dynamic` mode blends: `(1-w) * deltanet + w * (gqa + x)`

---

## Cross-Module Connections

### StateMetrics — The Unifying Interface

`StateMetrics` is the central data structure that connects all modules.
It flows from GatedDeltaNet to every other component:

```
                    GatedDeltaNet
                    ├── StateMetrics.frobenius_norm
                    ├── StateMetrics.mean_delta ──────────┐
                    └── StateMetrics.effective_rank       │
                                                          │
                  ┌───────────────────────────────────────┘
                  │
    ┌─────────────┼─────────────────┬─────────────────────┐
    ▼             ▼                 ▼                     ▼
MoeRouter    HybridAttention    Engram            [BlockDiffusion]
adaptive_k() routing_scores()  maybe_update()    block_boundaries()
  │               │                │                     │
  │ mean_delta    │ mean_delta     │ mean_delta ≥        │ mean_delta →
  │ → confidence  │ → routing      │   mdl_threshold     │   block size
  │ → reduce K    │   score boost  │   to store          │   adaptation
  │               │                │                     │
  ▼               ▼                ▼                     ▼
K=1..4        select tokens    add/skip entry     small/large blocks
              for GQA
```

### Dimension Alignment

All modules share consistent dimensions via their `chimere()` configs:

| Dimension      | Value | Used In                            |
|----------------|-------|------------------------------------|
| `hidden_dim`   | 4096  | All modules (input/output size)    |
| `num_heads`    | 32    | DeltaNet, GQA query heads          |
| `head_dim`     | 128   | DeltaNet state, Engram entry_dim   |
| `num_kv_heads` | 8     | GQA only (4 queries per KV group)  |
| `num_experts`  | 8     | MoE router (+ 1 shared)            |

The critical alignment: `head_dim == entry_dim == 128`. This means
DeltaNet per-head states and Engram entries live in the same space.

---

## Config Presets

Each module provides `chimere()`, `test()`, and sometimes other presets:

| Module          | Preset      | Key Parameters                     |
|-----------------|-------------|-------------------------------------|
| GatedDeltaNet   | `chimere()` | 4096d, 32h, 128hd, ChannelWise    |
| GatedDeltaNet   | `qwen3()`   | 4096d, 32h, 128hd, Scalar         |
| GatedDeltaNet   | `test()`    | 64d, 4h, 16hd, Scalar             |
| MoeRouter       | `chimere()` | 8 experts, K=1..4, q=2, shared    |
| MoeRouter       | `test()`    | 4 experts, K=1..3, q=2, shared    |
| Engram          | `chimere()` | dim=128, 4096 entries, mdl=0.1     |
| Engram          | `test()`    | dim=8, 32 entries, mdl=0.1         |
| HybridAttention | `chimere()` | Dynamic, 32q/8kv, cap=25%          |
| HybridAttention | `test()`    | Dynamic, 4q/2kv, cap=50%           |
| HybridStack     | `chimere_32()` | 32L, attn at 7,15,23,31        |
| HybridStack     | `qwen3_48()`   | 48L, attn every 4th            |

---

## Test Coverage

53 tests total, 0 warnings. All pass on CPU.

| Module              | Tests | What They Prove                                 |
|---------------------|-------|--------------------------------------------------|
| `lib.rs`            | 5     | Single/multi association, convergence (52% MDL reduction), decay gate, state metrics |
| `moe_router.rs`     | 10    | Tsallis entropy bounds, Shannon limit, adaptive K ± state, softmax, temperature, topK, Sinkhorn balance, E2E pipeline |
| `engram.rs`         | 11    | Poincaré distance/symmetry/hierarchy, depth ordering, Möbius in-ball, MDL gate, dedup, nearest query, compression, eviction, metrics |
| `hybrid_attention.rs` | 13  | 4 routing modes, capacity cap, sigmoid weights, GQA expansion (±identity), causal mask, token mask, stack configs (Chimère/Qwen3), GQA forward shapes, GQA with culling |
| `block_diffusion.rs` | 14   | 3 noise schedules (boundaries, monotonicity, comparison), reverse timesteps, forward noise (levels, preservation), denoise (confidence-based unmasking), full generation (perfect + noisy model), adaptive sizing (confident/uncertain), adaptive steps, multi-block AR, StateMetrics flow |

Key integration tests:
- **MDL convergence** (`lib.rs`): 10 cycles of ABCABC pattern → error drops 52%
- **State modulation** (`moe_router.rs`): confident DeltaNet state → K drops from 3 to 2
- **Compression** (`engram.rs`): repetitive input → 3 entries; novel input → 7+ entries
- **Capacity cap** (`hybrid_attention.rs`): 8 tokens, 25% cap → only 2 get GQA
- **StateMetrics → block sizing** (`block_diffusion.rs`): delta=0.05 → block≥112; delta=0.8 → block≤40
- **E2E generation** (`block_diffusion.rs`): perfect mock → 100% accuracy, 0 remaining masked

---

### 5. Block Diffusion (`block_diffusion.rs`)

The generation engine. Implements discrete masked diffusion following
BD3-LM / MDLM: autoregressive between blocks, parallel diffusion within.

#### Public Types

```rust
pub enum NoiseScheduleType { Cosine, Linear, Quadratic }

pub struct BlockDiffusionConfig {
    pub block_size: usize,         // 32 (bd3lm) / 64 (chimere)
    pub num_steps: usize,          // denoising steps per block
    pub schedule: NoiseScheduleType,
    pub vocab_size: usize,         // 151936 (Qwen2/Nanbeige)
    pub mask_token_id: u32,        // [MASK] token
    pub unmask_threshold: f32,     // min confidence to unmask
    pub greedy: bool,              // greedy vs nucleus sampling
    pub temperature: f32,
    pub adaptive_blocks: bool,     // entropy-adaptive block sizing
    pub min_block_size: usize,     // 16
    pub max_block_size: usize,     // 128
    pub entropy_boundary_threshold: f32, // 0.5
}

pub struct MaskedBlock {
    pub tokens: Vec<u32>,
    pub is_masked: Vec<bool>,
    pub clean_tokens: Option<Vec<u32>>,  // for training/testing
    pub noise_level: f64,
}

pub struct TokenPrediction {
    pub token_id: u32,
    pub confidence: f32,
    pub logits: Option<Vec<f32>>,
}

pub struct ScoreModelOutput {
    pub predictions: Vec<TokenPrediction>,
    pub state_metrics: Option<StateMetrics>,  // ← DeltaNet connection
}

pub struct GeneratedBlock {
    pub tokens: Vec<u32>,
    pub confidences: Vec<f32>,
    pub steps_used: usize,
    pub remaining_masked: usize,
    pub final_state_metrics: Option<StateMetrics>,  // ← for next block
    pub unmask_history: Vec<usize>,
}

pub trait ScoreModel {
    fn forward(&self, block: &MaskedBlock, context: &[u32]) -> ScoreModelOutput;
}
```

#### Public Functions

```rust
// Noise schedule
pub fn noise_level(t: f64, schedule: NoiseScheduleType) -> f64;
    // t ∈ [0,1], returns σ(t) ∈ [0,1]
    // Cosine: σ(t) = cos(π/2·(1-t))  — more aggressive at start
    // Linear: σ(t) = t
    // Quadratic: σ(t) = 1-(1-t)²

pub fn reverse_timesteps(num_steps: usize) -> Vec<(f64, f64)>;
    // Returns (t_current, t_next) pairs from t=1 to t=0

// Forward process (noising)
pub fn forward_noise(block: &MaskedBlock, target_noise: f64,
                     mask_token_id: u32, priority: Option<&[f64]>)
    -> MaskedBlock;
    // Masks target_noise fraction of tokens by priority

// Reverse process (denoising)
pub fn denoise_step(block: &MaskedBlock, scores: &ScoreModelOutput,
                    t_current: f64, t_next: f64, config: &BlockDiffusionConfig)
    -> MaskedBlock;
    // Unmasks highest-confidence positions based on schedule

// Full generation
pub fn generate_block(config: &BlockDiffusionConfig,
                      score_model: &dyn ScoreModel, context: &[u32])
    -> GeneratedBlock;
    // Full reverse process: fully masked → denoised block

pub fn generate_sequence(config: &BlockDiffusionConfig,
                         score_model: &dyn ScoreModel,
                         prompt: &[u32], num_blocks: usize)
    -> Vec<u32>;
    // Multi-block AR generation with adaptive sizing

// Entropy-adaptive (v2)
pub fn compute_adaptive_block_size(state_metrics: &StateMetrics,
                                    config: &BlockDiffusionConfig) -> usize;
    // Low delta → large blocks, high delta → small blocks
    // Rounded to multiples of 8

pub fn compute_adaptive_steps(block_size: usize) -> usize;
    // steps ∝ sqrt(block_size), clamped to [4, 32]

// Metrics
pub fn block_accuracy(generated: &[u32], clean: &[u32]) -> f64;
pub fn step_accuracy(block: &MaskedBlock) -> f64;
```

#### Key Invariants

- Noise schedule: σ(0)=0 (clean), σ(1)=1 (fully masked), monotonic
- Confidence-based unmasking: highest confidence tokens unmask first
- Already-unmasked positions are never re-masked during denoising
- Adaptive block sizes are multiples of 8 (tensor alignment)
- StateMetrics flows from score model → adaptive block sizing
- `generate_sequence` is block-autoregressive: block N context conditions block N+1

---

## What's NOT Here (and Where It Lives)

| Component                | Status  | Location / Notes                          |
|--------------------------|---------|-------------------------------------------|
| Training loop            | Python  | `dllm/` — BD3LM training via HuggingFace |
| Weight loading           | TODO    | Need safetensors/GGUF parser              |
| Tokenizer                | TODO    | `candle-transformers` has bindings         |
| CUDA kernels             | Blocked | Need CUDA 12.8+ for sm_120 (RTX 5060 Ti) |
| Expert FFN layers        | TODO    | SwiGLU FFNs, per-expert weight loading    |
| KV Cache (tiered)        | TODO    | L0:FP8/L1:INT4/L2:NTC/L3:Zstd            |
| Embedding / LM head      | TODO    | Tied embeddings, vocab_size projection    |
| Flash Attention          | TODO    | candle has partial FA2 bindings           |
| Quantization             | TODO    | Q4_K_M, INT8 for inference                |
| RoPE / position encoding | TODO    | Needed for GQA layers                     |

---

## Design Decisions

1. **StateMetrics as central bus**: Rather than ad-hoc signals between
   modules, StateMetrics provides a clean, typed interface. Every module
   reads the same 3 floats and interprets them for its own purpose.

2. **Standalone + Layer pattern**: Each module exposes both standalone
   functions (for testing, no weights) and full layers (with VarBuilder).
   This makes unit testing trivial while keeping the full layer realistic.

3. **Dynamic routing over fixed ratio**: Chimère doesn't hardcode which
   tokens get full attention. The router learns from the data which tokens
   need it. The capacity cap prevents pathological cases.

4. **Poincaré ball for Engram**: Hyperbolic space naturally encodes
   hierarchy without explicit tree structures. The exponential volume
   growth near the boundary provides room for specialization.

5. **Tsallis over Shannon**: q=2 Tsallis entropy is quadratic (cheaper
   than log) and penalizes dominant probabilities more strongly, giving
   sharper routing decisions.

6. **GQA over MHA**: 4 KV groups (32q/8kv) reduces KV cache by 4×
   with minimal quality loss. Critical for fitting in 16 GB VRAM.

---

# M1 Multi-slot + Continuous Batching (Apr 2026)

This section documents the `m1-multislot` serving path landed over
J1-J8 (2026-04-24). It describes the NEW code, not the legacy path —
for the pre-M1 single-slot hot path see §0 "Overview" above and
§"StateMetrics" cross-module diagram.

## Motivation

Before M1, `AppState.model: Mutex<AppStateModel>` serialised every
inference under a single `tokio::sync::Mutex` (`server.rs:313`). The
model Mutex is held for the *entire* generation (`blocking_lock`
across the `generate_with_mtp_streaming` SSE loop at `server.rs:927`),
and `llama_backend::kv_cache_seq_rm()` hard-codes `seq_id=0`
(`llama_backend.rs:1015-1017`). Result: **exactly one request in
flight at a time**, regardless of how many HTTP handlers the axum
runtime dispatches. Aggregate throughput under N≥2 concurrent clients
is ≈ single-request throughput, latency distribution is queue-
dominated.

M1 rewires the serving path to continuous batching (one `llama_decode`
per step, N logical slots, per-seq K/V + sampler + engram bias
isolation) while keeping the legacy Mutex path fully intact as the
default. The feature flag is `CHIMERE_MULTISLOT=<N>`: unset / `1`
selects the legacy path, `>= 2` routes through the new admission
queue + scheduler.

## Dataflow

```
      ┌──────────────────────────────────────────────────────────────┐
      │ HTTP clients (axum, N concurrent requests)                   │
      │   POST /v1/chat/completions                                  │
      └────────────────────────┬─────────────────────────────────────┘
                               │
                               │  ChatRequest → ScheduledRequest{
                               │      prompt, params, engram_hint,
                               │      tx: mpsc::Sender<StreamMsg>,
                               │      metadata: ScheduledRequestMeta }
                               ▼
      ┌──────────────────────────────────────────────────────────────┐
      │ AdmissionQueue: mpsc::Sender<ScheduledRequest>               │
      │   bounded at `ADMISSION_QUEUE_CAP` (64 by default)           │
      │   back-pressure via `send().await` when full                 │
      └────────────────────────┬─────────────────────────────────────┘
                               │
                               │  blocking_recv() from the one worker
                               ▼
      ┌──────────────────────────────────────────────────────────────┐
      │ SchedulerTask (1 OS thread, `chimere-sched-dispatch`)        │
      │                                                              │
      │   loop {                                                     │
      │     admit_new():                                             │
      │       drain rx; alloc_free Slot; init KV pages;              │
      │       park request if all slots busy                         │
      │                                                              │
      │     step():                                                  │
      │       build_decode_batch():                                  │
      │         for slot in active:                                  │
      │           Prefilling → push prefill chunk (no cross-seq      │
      │             in same batch — see J4 caveat below)             │
      │           Generating → push 1 tok @ slot.pos, logits=1       │
      │                                                              │
      │       forward_multi_seq(ctx, batch)                          │
      │         → one `llama_decode` call for the whole batch        │
      │         → returns (seq_id, logits[vocab_size]) for each      │
      │           entry with `request_logits=true`                   │
      │                                                              │
      │       per-slot sample:                                       │
      │         for slot in active:                                  │
      │           if !slot.thinking:                                 │
      │             apply_engram_bias_to_sampler()                   │
      │               = alpha * ln(prob + 1e-10), per-slot sampler   │
      │           (tok, lp) = sample_slot_with_logprobs(sampler,     │
      │                           ctx, slot.batch_idx)               │
      │           slot.accept_token(tok)                             │
      │           slot.tx.send(StreamMsg::{Token, Thinking, …})      │
      │           if slot.should_finish(tok): free_slot(slot)        │
      │   }                                                          │
      │                                                              │
      │   SlotPool = Vec<Slot>[NUM_SLOTS]                            │
      │   ├─ Slot 0: seq=0, sampler_0, engram_0, kv-pages seq 0      │
      │   ├─ Slot 1: seq=1, sampler_1, engram_1, kv-pages seq 1      │
      │   ├─ Slot 2: FREE (seq pre-assigned, pages empty)            │
      │   └─ Slot 3: FREE                                            │
      └────────────────────────┬─────────────────────────────────────┘
                               │
                               │  per-slot mpsc::Sender<StreamMsg>
                               │  (one channel per admitted request)
                               ▼
      ┌──────────────────────────────────────────────────────────────┐
      │ axum handlers — map StreamMsg → SSE `data:` frames           │
      │   { Token { text, logprob } } → `"delta":{"content":…}`      │
      │   { Thinking { text } }       → `"delta":{"reasoning_…":…}`  │
      │   { ToolCall { json } }       → `"delta":{"tool_calls":…}`   │
      │   { Done { finish_reason } }  → `"finish_reason":…`          │
      │   { Error { message } }       → 500 + SSE comment            │
      └──────────────────────────────────────────────────────────────┘
```

### Per-seq vs per-slot — where the data lives

| Entity                        | Per-seq   | Per-slot | Global (shared) |
|-------------------------------|:---------:|:--------:|:---------------:|
| KV cache pages (libllama)     |    X      |          |                 |
| SSM / GDN state (hybrid arch) |    X      |          |                 |
| `chimere_sampler` handle      |           |    X     |                 |
| logit_bias map                |           |    X     |                 |
| DRY / repetition history      |           |    X     |                 |
| engram **tables** (mmap)      |           |          |       X         |
| engram bias **applied**       |           |    X     |                 |
| tokenizer                     |           |          |       X         |
| model weights (GGUF mmap)     |           |          |       X         |
| `</think>` suppression `-inf` |           |    X     |                 |
| mpsc stream channel           |           |    X     |                 |
| `cancelled` atomic            |           |    X     |                 |

A `Slot` owns its `SamplerHandle` (a `*mut c_void` into
`chimere_sampler.cpp` allocated via `chimere_sampler_alloc_with_dry`).
The handle is `Send` but never `Clone`; the destructor calls
`chimere_sampler_free_handle`, so slot lifetime drives sampler
lifetime. Engram *tables* stay global (mmap'd `.engr` files, read-only
after boot) but the *biases derived from a lookup* are pushed into the
per-slot sampler via `chimere_sampler_set_engram_bias_handle`.

### Feature flag semantics

`SchedulerConfig::from_env()` reads `CHIMERE_MULTISLOT` and clamps to
`[1, NUM_SLOTS_MAX=8]`:

| Value     | `enabled` | Path                                                 |
|-----------|:---------:|------------------------------------------------------|
| unset     |  false    | Legacy `Mutex<AppStateModel>` — production default   |
| `0`, `1`  |  false    | Legacy path                                          |
| `2`       |  true     | 2 slots, admission queue routed                      |
| `3-8`     |  true     | N slots, capped at 8                                 |
| `≥ 9`     |  true     | Capped to 8, warning printed                         |

The scheduler is wired in `bin/chimere-server.rs:312-342`. When
disabled, `AppState.scheduler = None` and `AppState.multislot_active()
== false`; every handler takes the legacy path (see J4 note below for
the in-flight dispatcher rewrite).

## Module responsibilities

### `src/slot_scheduler.rs` (new, ~960 lines)

- `SchedulerConfig` + `Scheduler` + `SlotPool` + `Slot` + `BatchBuilder`.
- `ScheduledRequest` holds a `Box<dyn FnOnce(ScheduledRequestMeta) + Send>`
  closure so the scheduler is decoupled from `chimere_model::*`.
  `ScheduledRequestMeta` carries `request_id`, prompt token count,
  `cancelled` flag (client-disconnect signalling), and `enqueued_at`
  for queue-wait telemetry.
- `SlotState` = `Free | Prefilling { chunks_done } | Generating |
  Draining`. State transitions live on `Slot`, not on an external
  FSM — keeps invariants local.
- `Slot::apply_engram_bias_to_sampler()` — implements the production
  formula `alpha * ln(prob + eps)` identically to `mtp_scheduler.rs`
  so the multi-slot path is numerically equivalent to the single-slot
  path on the same prompt.
- `SlotPool::alloc_samplers_with_dry()` — allocates N independent
  `chimere_sampler` via `chimere_sampler_alloc_with_dry`, rolling back
  to "no sampler" if any slot fails. Called once at scheduler boot.
- `SamplerHandle` / `EngramHandle` — owning FFI wrappers.
  `SamplerHandle` is `Send` but `!Sync` and `!Clone`; `EngramHandle`
  is `Clone` (wraps `Arc<MultiEngramLookup>`).

### `src/llama_backend.rs` (J3 additions, ~210 lines)

- `LlamaForward::forward_multi_seq(&mut self, entries: &[MultiSeqEntry])
  -> Result<Vec<(i32, Vec<f32>)>, String>` — single `llama_decode`
  call composing N seq_ids in one batch. Returns per-entry logits for
  entries that requested them.
- `MultiSeqEntry { token, pos, seq_id, request_logits }` — input shape.
- `LlamaForward::kv_cache_seq_rm_for(seq_id) -> bool` — releases KV
  pages owned by a finished seq. The legacy call site at line 1015
  (`kv_cache_seq_rm(-1, -1, -1)`) still exists for the single-slot
  path.
- `LlamaForward::sample_slot_with_logprobs(sampler, idx)` — per-slot
  sample + logprobs from a given batch index (used by the scheduler's
  per-slot sample step).
- FFI extern declarations for
  `chimere_sampler_alloc_with_dry`,
  `chimere_sampler_set_engram_bias_handle`,
  `chimere_sampler_set_logit_bias_handle`,
  `chimere_sampler_clear_engram_bias_handle`,
  `chimere_sampler_reset_handle`,
  `chimere_sampler_free_handle`.

### `ffi/chimere_sampler.cpp` (J5 rewrite, ~290 lines)

Rewritten from the libcommon-based sampler (`common_sampler_init`,
crashed at startup on 2026-04-24 due to an ABI drift in
`struct common_sampler`) to a minimal libllama-only chain. See
`~/Bureau/chimere-sampler-unblock-2026-04-24.md` for the full
investigation.

- `struct chimere_sampler` — per-instance state: sampling knobs
  (`temperature`, `top_p`, `top_k`, `min_p`, presence/DRY params),
  a `std::unordered_map<llama_token,float>` for the logit_bias map,
  a `std::vector<llama_token>` for `prev` history, and a
  `std::mt19937` RNG.
- `sample_chain()` — `repetition penalties → top-k → top-p → min-p
  → temperature → llama_sample_token`. Greedy when `temp ≤ 0`.
- `chimere_sampler_set_engram_bias_handle(token_ids, biases, n)` —
  installs the engram-derived biases on top of any existing logit
  biases. Preserves manual `-inf` biases (e.g. `</think>` suppression
  from the `generate.rs` path).
- `chimere_sampler_set_logit_bias_handle` / `_clear_engram_bias_handle`
  — auxiliary ABI for tests and the scheduler's slot free path.
- No dependency on `libcommon.a`; the `ffi/build.rs` link line was
  pruned accordingly.

### `src/server.rs` (J2 touch, ~30 lines)

- `AppState` grows `pub scheduler: Option<Arc<Scheduler>>` and a
  `multislot_active()` helper. Single-slot callers ignore the field.
- `bin/chimere-server.rs` builds the scheduler iff
  `SchedulerConfig::from_env().is_active()`, spawns the dispatcher
  thread via `Scheduler::spawn_workers()`, and detaches the JoinHandle
  (dispatcher is process-lifetime).

## State machine per slot

```
          ┌──────┐
          │ Free │ ←────────────────────────────────────────┐
          └──┬───┘                                          │
             │ admit_new: prompt tokenised, KV pages        │
             │ pre-reserved, engram alpha bound             │
             ▼                                              │
   ┌────────────────────────┐                               │
   │ Prefilling {           │                               │
   │   chunks_done: 0..K    │                               │
   │ }                      │                               │
   └──┬─────────────────────┘                               │
      │ last prefill chunk accepted, logits requested       │
      ▼                                                     │
   ┌────────────┐                                           │
   │ Generating │──────────┐                                │
   └──┬─────────┘          │ stop-token / max_tokens        │
      │ client disconnect  │ hit → schedule Draining        │
      │ (cancelled=true)   │                                │
      ▼                    ▼                                │
   ┌────────────┐    ┌──────────┐                           │
   │  Draining  │◄───│ Generating│ (state unchanged, emits  │
   │ (one step) │    └──────────┘  Done on next step)       │
   └──┬─────────┘                                           │
      │ Done marker emitted, KV pages released via          │
      │ llama_kv_cache_seq_rm(ctx, seq_id, 0, -1),          │
      │ sampler.clear_engram_bias() + .reset(),             │
      │ SlotPool::free_notify.notify_one()                  │
      └─────────────────────────────────────────────────────┘
```

## Batch construction invariants

From the J4 smoke (`bin/j4_smoke.rs`), ik_llama's `qwen3next` path
**does not** accept mixed prefill + generate for the same seq_id in
the same `llama_decode` batch — it logs
`qwen3next mixed-sequence batch contains repeated seq_id values;
falling back to single-token chunking` and reorders the batch,
breaking the isolated-baseline equivalence we rely on.

The scheduler's `BatchBuilder` therefore honours two invariants per
call:

1. **No repeated seq_id within one `llama_decode` batch.** A slot
   either contributes its next prefill chunk OR one generate token,
   never both in the same step.
2. **Logits requested only for the tokens we will sample.** The
   `logits: Vec<i8>` flag is `1` for the last prefill token (so the
   first generate step has something to sample from) and for every
   generate step; it is `0` for all other prefill tokens.

Cross-seq mixing is fine — a batch of `{A prefill chunk 512 tokens, B
gen 1 token, C gen 1 token}` works as long as no seq_id appears more
than once. J4 smoke proves seq-1's token stream is bit-for-bit
identical whether it runs alone or interleaved with a 512-token
prefill of seq-0.

## MTP gating

MTP (multi-token prediction speculative decoding, `mtp_scheduler.rs`)
mutates `mtp_op_type` on the *context-wide* `llama_forward` state, not
per-seq. A multi-slot MTP would require per-seq MTP state that
ik_llama does not (yet) support. Policy:

- `CHIMERE_MULTISLOT >= 2` and **any** slot wants MTP → scheduler
  routes that slot into a "slot-exclusive" mode: no other slot may be
  Generating while MTP is active, and the batch falls back to
  single-seq for the duration.
- Default: **MTP disabled** when `CHIMERE_MULTISLOT >= 2`, a warning
  is logged at boot. J6 will finalise the slot-exclusive mechanism;
  until then, the multi-slot path generates without MTP speculation
  and accepts the ~5-8% throughput loss vs single-slot MTP.
- The single-slot path (`CHIMERE_MULTISLOT` unset / `1`) is unchanged
  — MTP stays armed via the legacy `generate_with_mtp_streaming`.

## Status (J8, 2026-04-24)

| Layer                                   | Status | Smoke                     |
|-----------------------------------------|:------:|---------------------------|
| `slot_scheduler.rs` types + admission   |  OK    | `j2-smoke` PASS           |
| `forward_multi_seq` FFI                 |  OK    | `j3-smoke` PASS           |
| chunked prefill + concurrent gen        |  OK    | `j4-smoke` PASS           |
| per-slot sampler + engram isolation     |  OK    | `j5-smoke` PASS (Apr 24)  |
| HTTP dispatcher → `forward_multi_seq`   |  WIP   | deferred from J4 brief    |
| stop / cancel / disconnect cleanup      |  PENDING | J6                      |
| stress (4c, 8 backlog, 1000-req leak)   |  PENDING | J7                      |
| bench harness                           |  OK    | `bench-m1` + `scripts/bench_m1.sh` |

Production `:8081` is not touched by any of this — the legacy path is
byte-for-byte unchanged for `CHIMERE_MULTISLOT` unset. The M1 path
activates only when the env var is explicitly set to `>= 2`.

## Files

- `chimere-server/src/slot_scheduler.rs` — scheduler + slot + FFI
  handles (~960 lines, created J1, extended J2/J5)
- `chimere-server/src/llama_backend.rs:1015+` — `forward_multi_seq`,
  `kv_cache_seq_rm_for`, `sample_slot_with_logprobs` (added J3/J5)
- `chimere-server/ffi/chimere_sampler.cpp` — libllama-only sampler
  chain (rewritten J5)
- `chimere-server/src/server.rs:308+` — `AppState.scheduler`
- `chimere-server/src/bin/chimere-server.rs:312` — scheduler wiring
- `chimere-server/src/bin/{j2,j3,j4,j5}_smoke.rs` — per-step smokes
- `chimere-server/src/bin/bench_m1.rs` — J8 bench harness
- `scripts/bench_m1.sh` — concurrency sweep wrapper
