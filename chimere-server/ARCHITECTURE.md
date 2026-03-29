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
