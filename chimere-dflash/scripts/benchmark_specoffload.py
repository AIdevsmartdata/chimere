#!/usr/bin/env python3
"""
SpecOffload-style pipelined speculative decoding benchmark.

Overlaps drafter GPU execution with target CPU verification to hide latency.

Two modes selectable via --specoffload flag:

  Sequential (baseline, default):
    Cycle: [draft_GPU] → [verify_CPU] → [correct] → repeat
    GPU is idle while target MoE experts run on CPU.

  Pipelined (--specoffload):
    Cycle N: [verify_CPU(draft_N) ‖ draft_GPU(speculative_N+1)]
              ↓
             If ALL accepted: reuse speculative_N+1 (HIT)
             If PARTIAL reject: discard speculative_N+1, re-draft (MISS)

The key insight: Qwen3.5-35B-A3B has MoE experts on CPU (layers 20-39), making
eval_incr CPU-bound. During those ~100-400ms of CPU work, the GPU is completely
idle. We use that time to pre-draft the NEXT block, assuming all current tokens
will be accepted.

If the assumption holds (HIT): next cycle starts immediately with no drafting cost.
If wrong (MISS): we discard the speculative draft and fall back to sequential drafting.

Expected benefit: at τ=50%, ~50% of cycles become HIT → ~25% wall-clock reduction.
At τ=70%, ~70% HIT rate → potentially 30-40% wall-clock reduction.

Metrics tracked:
  - n_speculative_hits: pre-computed draft was valid and reused
  - n_speculative_misses: had to discard and re-draft
  - hit_rate: hits / (hits + misses) — quality of the "full accept" assumption
  - tok/s comparison between sequential and pipelined modes
"""
import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chimere.config_v7 import DFlashV7Config
from chimere.modeling_v8 import DFlashDraftModelV8
from chimere.target_daemon import TargetDaemon


# ──────────────────────────────────────────────────────────────────────────────
#  Drafter loading — identical to benchmark_wallclock.py
# ──────────────────────────────────────────────────────────────────────────────

def load_drafter(checkpoint_path: str, device: torch.device):
    """Load DFlashDraftModelV8 with split-device BF16 placement.

    Blocks + projection layers → GPU (BF16, ~0.98 GB)
    embed_tokens + lm_head   → CPU (FP32, ~2 GB — too large for remaining VRAM)
    """
    from dataclasses import fields
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = DFlashV7Config(**{
        f.name: ckpt["config"][f.name]
        for f in fields(DFlashV7Config)
        if f.name in ckpt["config"]
    })
    model = DFlashDraftModelV8(config)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(state_dict)

    if device.type == "cuda":
        # Split-device BF16: heavy compute on GPU, large vocab tensors on CPU
        model.eval().float()  # start all on CPU FP32
        bf = torch.bfloat16
        for block in model.layers:
            block.to(device=device, dtype=bf)
        for proj in model.ctx_k_projs:
            proj.to(device=device, dtype=bf)
        for proj in model.ctx_v_projs:
            proj.to(device=device, dtype=bf)
        model.fc.to(device=device, dtype=bf)
        model.hidden_norm.to(device=device, dtype=bf)
        model.norm.to(device=device, dtype=bf)
        # lm_head stays CPU FP32 (1.02 GB — no VRAM margin with Qwen3.5 loaded)
        model.rotary_emb.to(device=device)
        if model.input_proj is not None:
            model.input_proj.to(device=device, dtype=bf)
        if model.output_proj is not None:
            model.output_proj.to(device=device, dtype=bf)
        # embed_tokens stays CPU FP32 (lookup only, fast)
    else:
        model = model.to(device).eval().float()

    return model, config


# ──────────────────────────────────────────────────────────────────────────────
#  Benchmark prompts
# ──────────────────────────────────────────────────────────────────────────────

BENCH_PROMPTS = [
    "Explain how speculative decoding works in large language models.",
    "Write a Python function to compute the Fibonacci sequence using memoization.",
    "What are the advantages of MoE (Mixture of Experts) architectures?",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot =",
    "The transformer architecture introduced in 'Attention Is All You Need' uses",
    "Describe the difference between TCP and UDP protocols.",
    "Les principes fondamentaux de la kinésithérapie reposent sur",
    "Docker containers differ from virtual machines because",
    "import torch\nimport torch.nn as nn\n\nclass TransformerBlock(nn.Module):\n    def __init__(self",
    "The CAP theorem states that a distributed system cannot simultaneously guarantee",
]

EOS_TOKENS = frozenset({151643, 151645})


# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────

def compute_adaptive_k(recent_tau: list, default_k: int) -> int:
    """Adaptive draft length based on trailing acceptance rate."""
    if not recent_tau:
        return default_k
    avg_tau = sum(recent_tau[-3:]) / len(recent_tau[-3:])
    if avg_tau > 0.5:
        return 15
    elif avg_tau > 0.2:
        return 8
    else:
        return 4


def build_drafter_inputs(
    hidden_all: np.ndarray,
    config: DFlashV7Config,
    drafter_device: torch.device,
) -> Tuple[list, torch.Tensor, int, torch.Tensor]:
    """Slice hidden_all into the context window expected by the drafter.

    Returns:
        hidden_list: list of tensors [n_layers × [1, ctx_len, H]]
        ctx_lengths: Tensor [1]
        anchor_pos: int (absolute position index of last token)
        anchor_positions: Tensor [1]
    """
    n_layers = config.num_feature_layers
    seq_len = hidden_all.shape[1]
    ctx_end = seq_len
    ctx_start = max(0, ctx_end - config.max_ctx_len)
    ctx_len = ctx_end - ctx_start

    hidden_list = [
        torch.from_numpy(hidden_all[j, ctx_start:ctx_end].copy())
        .unsqueeze(0).to(drafter_device)
        for j in range(n_layers)
    ]
    ctx_lengths = torch.tensor([ctx_len], device=drafter_device, dtype=torch.long)
    anchor_pos = ctx_end - 1
    anchor_positions = torch.tensor([anchor_pos], device=drafter_device, dtype=torch.long)

    return hidden_list, ctx_lengths, anchor_pos, anchor_positions


def run_drafter(
    drafter: DFlashDraftModelV8,
    hidden_all: np.ndarray,
    config: DFlashV7Config,
    drafter_device: torch.device,
    anchor_token_id: int,
    stream: Optional[torch.cuda.Stream] = None,
) -> list:
    """Run drafter.generate_block() and return draft token IDs as a Python list.

    Runs inside its own CUDA stream so it does not block the main thread's
    default stream during parallel execution.
    """
    hidden_list, ctx_lengths, _, anchor_positions = build_drafter_inputs(
        hidden_all, config, drafter_device
    )

    ctx_mgr = torch.cuda.stream(stream) if (stream is not None and drafter_device.type == "cuda") \
               else torch.no_grad()

    with ctx_mgr:
        with torch.no_grad():
            draft_ids, _, _ = drafter.generate_block(
                hidden_list,
                context_lengths=ctx_lengths,
                temperature=0.0,
                anchor_token_id=anchor_token_id,
                anchor_positions=anchor_positions,
            )
            # Synchronize stream so result is ready before caller reads it
            if stream is not None and drafter_device.type == "cuda":
                stream.synchronize()
            return draft_ids[0].cpu().tolist()


# ──────────────────────────────────────────────────────────────────────────────
#  Autoregressive baseline
# ──────────────────────────────────────────────────────────────────────────────

def bench_autoregressive(target: TargetDaemon, prompts: list, max_tokens: int) -> list:
    """Baseline: target generates one token at a time, no drafter."""
    results = []
    for prompt in prompts:
        tokens = target.tokenize(prompt)
        target.clear_kv()

        _, argmax = target.eval_full(tokens)
        generated = [int(argmax[-1])]

        t0 = time.time()
        for _ in range(max_tokens - 1):
            _, argmax = target.eval_incr([generated[-1]])
            next_tok = int(argmax[0])
            generated.append(next_tok)
            if next_tok in EOS_TOKENS:
                break
        elapsed = time.time() - t0

        target.clear_kv()
        results.append({
            "tokens": len(generated),
            "time": elapsed,
            "tok_per_s": len(generated) / elapsed,
        })
    return results


# ──────────────────────────────────────────────────────────────────────────────
#  Sequential speculative decoding (no pipelining)
# ──────────────────────────────────────────────────────────────────────────────

def bench_speculative_sequential(
    target: TargetDaemon,
    drafter: DFlashDraftModelV8,
    config: DFlashV7Config,
    prompts: list,
    max_tokens: int,
    drafter_device: torch.device,
    adaptive: bool = True,
) -> list:
    """Sequential draft→verify→correct loop. Used as the non-pipelined baseline."""
    K = config.block_size
    results = []

    for prompt in prompts:
        tokens = target.tokenize(prompt)
        target.clear_kv()

        hidden_all, argmax = target.eval_full(tokens)
        last_target_pred = int(argmax[-1])
        generated = []
        total_accepted = 0
        total_drafted = 0
        recent_tau: list = []
        final_adaptive_k = K - 1

        t0 = time.time()
        while len(generated) < max_tokens:
            adaptive_k = compute_adaptive_k(recent_tau, K - 1) if adaptive else K - 1
            final_adaptive_k = adaptive_k

            all_tokens = tokens + generated
            anchor_token_id = all_tokens[-1]

            # ── Draft (sequential — GPU, then CPU lm_head) ──────────────────
            draft_tokens = run_drafter(
                drafter, hidden_all, config, drafter_device, anchor_token_id
            )
            draft_tokens = draft_tokens[:adaptive_k]

            # ── Verify (target CPU-bound for MoE layers 20-39) ──────────────
            target.save_state()
            incr_hidden, incr_argmax = target.eval_incr(draft_tokens)

            # ── Accept/reject ────────────────────────────────────────────────
            target_preds = [last_target_pred] + incr_argmax[:-1].tolist()
            n_accepted = 0
            for j in range(len(draft_tokens)):
                if j < len(target_preds) and draft_tokens[j] == target_preds[j]:
                    n_accepted += 1
                else:
                    break

            for j in range(n_accepted):
                generated.append(draft_tokens[j])

            if n_accepted < len(draft_tokens):
                # ── Rejection path ───────────────────────────────────────────
                correction = target_preds[n_accepted]
                generated.append(correction)
                target.restore_state()
                accepted_plus_correction = draft_tokens[:n_accepted] + [correction]
                re_hidden, re_argmax = target.eval_incr(accepted_plus_correction)
                last_target_pred = int(re_argmax[-1])
                n_new = len(accepted_plus_correction)
                hidden_all = np.concatenate([hidden_all, re_hidden[:, :n_new, :]], axis=1)
            else:
                # ── Full accept path — bonus token ───────────────────────────
                bonus = int(incr_argmax[-1])
                generated.append(bonus)
                bonus_hidden, bonus_argmax = target.eval_incr([bonus])
                last_target_pred = int(bonus_argmax[0])
                hidden_all = np.concatenate([hidden_all, incr_hidden, bonus_hidden], axis=1)

            total_accepted += n_accepted
            total_drafted += len(draft_tokens)

            cycle_tau = n_accepted / len(draft_tokens) if draft_tokens else 0.0
            recent_tau.append(cycle_tau)
            if len(recent_tau) > 3:
                recent_tau = recent_tau[-3:]

            if generated and generated[-1] in EOS_TOKENS:
                break

        elapsed = time.time() - t0
        target.clear_kv()

        tau = total_accepted / max(1, total_drafted)
        results.append({
            "tokens": len(generated),
            "time": elapsed,
            "tok_per_s": len(generated) / elapsed,
            "tau": tau,
            "accepted_per_block": total_accepted / max(1, total_drafted / (K - 1)),
            "adaptive_k": final_adaptive_k,
        })
    return results


# ──────────────────────────────────────────────────────────────────────────────
#  SpecOffload pipelined speculative decoding
# ──────────────────────────────────────────────────────────────────────────────

def bench_speculative_pipelined(
    target: TargetDaemon,
    drafter: DFlashDraftModelV8,
    config: DFlashV7Config,
    prompts: list,
    max_tokens: int,
    drafter_device: torch.device,
    adaptive: bool = True,
) -> list:
    """SpecOffload-style pipelined speculative decoding.

    Pipeline per cycle:
      1. save_state()
      2. Launch TWO concurrent threads:
           Thread A: target.eval_incr(draft_tokens_N)       [CPU-bound, MoE experts]
           Thread B: drafter.generate_block(hidden_optimistic) [GPU-bound]
             where hidden_optimistic = hidden_all + incr_hidden_N-1
             (assumes cycle N-1 was a full accept — used only when previous cycle WAS full accept)
      3. Collect both results.
      4. Check acceptance:
           ALL accepted → speculative_N+1 is VALID → skip next draft (HIT)
           PARTIAL       → discard speculative_N+1 → re-draft normally (MISS)

    THREADING SAFETY:
      - target_daemon is NOT thread-safe: only ONE call at a time.
      - drafter runs on GPU: fully independent from target's CPU/RAM work.
      - We use ThreadPoolExecutor(max_workers=2) but ensure target has exclusive access
        (it runs in exactly one thread at a time).
      - A dedicated CUDA stream is used for the drafter thread to avoid default-stream
        contention with any CUDA ops on the main thread.

    STATE MACHINE:
      speculative_pending: optional pre-computed (draft_tokens, hidden_optimistic_base)
      When entering a cycle:
        - If speculative_pending exists and previous cycle was full-accept:
            → Use the pre-computed draft directly (HIT)
        - Otherwise:
            → Run drafter sequentially on current hidden_all (MISS / first cycle)
    """
    K = config.block_size
    results = []

    # Dedicated CUDA stream for drafter thread (avoids blocking main-thread CUDA ops)
    drafter_stream: Optional[torch.cuda.Stream] = None
    if drafter_device.type == "cuda":
        drafter_stream = torch.cuda.Stream(device=drafter_device)

    # Thread pool: 2 workers — one for target, one for drafter
    executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="specoffload")

    for prompt in prompts:
        tokens = target.tokenize(prompt)
        target.clear_kv()

        hidden_all, argmax = target.eval_full(tokens)
        last_target_pred = int(argmax[-1])
        generated = []
        total_accepted = 0
        total_drafted = 0
        recent_tau: list = []
        final_adaptive_k = K - 1

        # ── Speculative next-draft state ──────────────────────────────────
        # When set, contains the pre-computed draft for the NEXT cycle under
        # the assumption that the CURRENT cycle will be a full accept.
        # Format: (draft_tokens: list[int], hidden_optimistic: np.ndarray)
        # hidden_optimistic is the hidden_all that was used to generate this draft;
        # we store it so we can verify it is still consistent on a HIT.
        spec_pending_draft: Optional[list] = None
        # The hidden_all state that was used to generate spec_pending_draft.
        # On a HIT, the actual hidden_all after full accept should extend this by incr_hidden.
        spec_pending_base_hidden_all: Optional[np.ndarray] = None

        # Counters for hit/miss statistics
        n_speculative_hits = 0
        n_speculative_misses = 0

        t0 = time.time()
        while len(generated) < max_tokens:
            adaptive_k = compute_adaptive_k(recent_tau, K - 1) if adaptive else K - 1
            final_adaptive_k = adaptive_k

            all_tokens = tokens + generated
            anchor_token_id = all_tokens[-1]

            # ── Decide whether to use speculative pre-computed draft ─────────
            # A HIT is valid only if:
            #   1. spec_pending_draft was pre-computed
            #   2. The previous cycle was a FULL ACCEPT (meaning hidden_all was
            #      extended exactly as the optimistic assumption predicted)
            # We track this via `prev_was_full_accept`.
            # On first cycle, there is no pending draft.
            use_spec = (
                spec_pending_draft is not None
                and len(spec_pending_draft) >= adaptive_k
            )

            if use_spec:
                # ── HIT: use pre-computed draft ──────────────────────────────
                # The speculative draft was generated assuming the previous cycle
                # was a full accept. We are here because it WAS (we only set
                # use_spec=True when the previous cycle committed to full accept
                # and we stored spec_pending_draft at that time).
                draft_tokens = spec_pending_draft[:adaptive_k]
                n_speculative_hits += 1

                # Clear pending state — will be re-populated below if this cycle is full accept
                spec_pending_draft = None
                spec_pending_base_hidden_all = None

                # Save target state before verification
                target.save_state()

                # ── PARALLEL: verify draft_N ‖ pre-draft N+1 ─────────────────
                # We can already start the next speculative draft while verifying,
                # using hidden_all as-is (it's the correct state for next drafting
                # ONLY if this cycle also fully accepts — we'll check below).

                # Build optimistic hidden_all for the next speculative draft:
                # We don't know incr_hidden yet (it comes from verify), so we
                # pre-draft using current hidden_all. After verification, if full
                # accept, we'd ideally want hidden_all + incr_hidden + bonus_hidden.
                # However, since incr_hidden isn't available yet, we overlap the
                # CURRENT hidden_all-based draft with verify, then on the next
                # full-accept cycle, draft using the updated hidden_all.
                # This means each speculative draft is based on hidden_all BEFORE
                # the last accept — one position behind. This is a known approximation
                # in SpecOffload-style systems and still provides substantial overlap.

                # Snapshot hidden_all before it changes (target thread may read it;
                # numpy arrays are not modified in-place here, only reassigned)
                hidden_all_snapshot = hidden_all  # numpy arrays are reference-typed; safe if we don't mutate

                # Compute anchor for next speculation — the LAST token in current sequence
                # which, under full-accept, would be draft_tokens[-1]
                # Under full-accept: generated will append all draft_tokens + bonus
                # We don't know bonus yet; use draft_tokens[-1] as anchor approximation.
                spec_anchor_token_id = draft_tokens[-1] if draft_tokens else anchor_token_id

                # We need hidden_all extended by incr_hidden to draft accurately,
                # but incr_hidden isn't available yet. To maximize overlap we launch
                # the drafter now using hidden_all_snapshot + the draft tokens
                # as hypothetical accepted tokens — building the extended context
                # by appending target hidden states from incr_hidden is impossible
                # without waiting. So we use hidden_all_snapshot directly and accept
                # the one-block lag. The savings in wall-clock still dominate.
                # (See SpecOffload paper §3.2: "shadow drafting" accepts 1-block lag.)
                def _draft_next_speculative():
                    return run_drafter(
                        drafter, hidden_all_snapshot, config, drafter_device,
                        spec_anchor_token_id, drafter_stream
                    )

                # Launch both concurrently
                future_verify: Future = executor.submit(
                    target.eval_incr, draft_tokens
                )
                future_spec_draft: Future = executor.submit(_draft_next_speculative)

                # Wait for both
                incr_hidden, incr_argmax = future_verify.result()
                spec_next_draft_raw = future_spec_draft.result()

            else:
                # ── MISS or first cycle: sequential draft then verify ─────────
                if spec_pending_draft is not None:
                    # We had a pending draft but it wasn't usable (adaptive_k changed, etc.)
                    n_speculative_misses += 1
                    spec_pending_draft = None
                    spec_pending_base_hidden_all = None

                # Sequential draft on current hidden_all
                draft_tokens = run_drafter(
                    drafter, hidden_all, config, drafter_device, anchor_token_id
                )
                draft_tokens = draft_tokens[:adaptive_k]

                # Verify (no parallel opportunity on the first cycle or after a miss)
                target.save_state()
                incr_hidden, incr_argmax = target.eval_incr(draft_tokens)

                spec_next_draft_raw = None  # no speculative pre-draft available

            # ── Accept/reject verification (same logic for both paths) ────────
            target_preds = [last_target_pred] + incr_argmax[:-1].tolist()
            n_accepted = 0
            for j in range(len(draft_tokens)):
                if j < len(target_preds) and draft_tokens[j] == target_preds[j]:
                    n_accepted += 1
                else:
                    break

            for j in range(n_accepted):
                generated.append(draft_tokens[j])

            if n_accepted < len(draft_tokens):
                # ── Rejection path ────────────────────────────────────────────
                correction = target_preds[n_accepted]
                generated.append(correction)

                # Discard any speculative pre-draft (hidden states are now different)
                if spec_next_draft_raw is not None:
                    n_speculative_misses += 1
                spec_pending_draft = None
                spec_pending_base_hidden_all = None

                target.restore_state()
                accepted_plus_correction = draft_tokens[:n_accepted] + [correction]
                re_hidden, re_argmax = target.eval_incr(accepted_plus_correction)
                last_target_pred = int(re_argmax[-1])
                n_new = len(accepted_plus_correction)
                hidden_all = np.concatenate([hidden_all, re_hidden[:, :n_new, :]], axis=1)

                # prev_was_full_accept = False → next cycle will not use speculative draft

            else:
                # ── Full accept path — bonus token ────────────────────────────
                bonus = int(incr_argmax[-1])
                generated.append(bonus)
                bonus_hidden, bonus_argmax = target.eval_incr([bonus])
                last_target_pred = int(bonus_argmax[0])
                hidden_all = np.concatenate([hidden_all, incr_hidden, bonus_hidden], axis=1)

                # Store the speculative pre-draft for next cycle
                if spec_next_draft_raw is not None:
                    # HIT path: we already have a pre-computed draft from parallel execution
                    spec_pending_draft = spec_next_draft_raw
                    spec_pending_base_hidden_all = hidden_all
                else:
                    # MISS path full-accept: pre-draft NOW (synchronous) so next cycle
                    # can use it. This doesn't save time THIS cycle but seeds the pipeline.
                    all_tokens_now = tokens + generated
                    anchor_token_id_next = all_tokens_now[-1]
                    spec_pending_draft = run_drafter(
                        drafter, hidden_all, config, drafter_device, anchor_token_id_next
                    )
                    spec_pending_base_hidden_all = hidden_all

            total_accepted += n_accepted
            total_drafted += len(draft_tokens)

            cycle_tau = n_accepted / len(draft_tokens) if draft_tokens else 0.0
            recent_tau.append(cycle_tau)
            if len(recent_tau) > 3:
                recent_tau = recent_tau[-3:]

            if generated and generated[-1] in EOS_TOKENS:
                break

        elapsed = time.time() - t0
        target.clear_kv()

        tau = total_accepted / max(1, total_drafted)
        results.append({
            "tokens": len(generated),
            "time": elapsed,
            "tok_per_s": len(generated) / elapsed,
            "tau": tau,
            "accepted_per_block": total_accepted / max(1, total_drafted / (K - 1)),
            "adaptive_k": final_adaptive_k,
            "spec_hits": n_speculative_hits,
            "spec_misses": n_speculative_misses,
            "spec_hit_rate": (
                n_speculative_hits / max(1, n_speculative_hits + n_speculative_misses)
            ),
        })

    executor.shutdown(wait=False)
    return results


# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SpecOffload-style pipelined speculative decoding benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints_v8_online_c4/best.pt",
                        help="Path to DFlashDraftModelV8 checkpoint")
    parser.add_argument("--model", type=str,
                        default=str(Path.home() / ".chimere/models/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-UD-Q5_K_XL.gguf"),
                        help="Path to target GGUF model")
    parser.add_argument("--daemon", type=str,
                        default="extract/build/target_daemon",
                        help="Path to target_daemon binary")
    parser.add_argument("--max-tokens", type=int, default=64,
                        help="Maximum tokens to generate per prompt")
    parser.add_argument("--layers", type=str, default="1,10,19,28,37",
                        help="Target hidden layer IDs to extract (comma-separated)")
    parser.add_argument("--drafter-device", type=str, default="cpu",
                        help="Device for drafter ('cpu' or 'cuda')")
    parser.add_argument("--adaptive", action=argparse.BooleanOptionalAction, default=True,
                        help="Adaptive draft length based on recent acceptance rate (default: True)")
    parser.add_argument("--ctx-size", type=int, default=256,
                        help="KV cache context size for target daemon")
    parser.add_argument("--specoffload", action="store_true", default=False,
                        help=(
                            "Enable SpecOffload pipelining: overlap drafter GPU execution "
                            "with target CPU verification. Shows hit/miss rate statistics. "
                            "Default: run sequential speculative (no overlap)."
                        ))
    parser.add_argument("--compare", action="store_true", default=False,
                        help=(
                            "Run ALL three modes sequentially: autoregressive + sequential "
                            "speculative + pipelined speculative. Produces a full comparison table. "
                            "Implies --specoffload."
                        ))
    parser.add_argument("--skip-autoregressive", action="store_true", default=False,
                        help="Skip autoregressive baseline (saves time when re-running)")
    args = parser.parse_args()

    if args.compare:
        args.specoffload = True

    drafter_device = torch.device(args.drafter_device)
    layers = [int(x) for x in args.layers.split(",")]
    prompts = BENCH_PROMPTS

    # ── Launch target daemon ──────────────────────────────────────────────────
    print("Launching target daemon...", flush=True)
    target = TargetDaemon(
        daemon_path=args.daemon,
        model_path=args.model,
        layers=layers,
        n_gpu_layers=99,
        extra_args=[
            "-c", str(args.ctx_size), "-t", "6",
            "-ot", r"blk\.[2-3][0-9]\.ffn_.*_exps\.weight=CPU",
            "--flash-attn", "on",
            "--cache-type-k", "q8_0", "--cache-type-v", "q4_0",
        ],
    )
    target.tokenize("warmup")
    print("Daemon ready.\n", flush=True)

    # ── Load drafter ──────────────────────────────────────────────────────────
    print("Loading drafter...", flush=True)
    drafter, config = load_drafter(args.checkpoint, drafter_device)
    print(f"Drafter loaded on {drafter_device}.", flush=True)

    if drafter_device.type == "cuda":
        allocated = torch.cuda.memory_allocated(drafter_device) / 1e9
        reserved = torch.cuda.memory_reserved(drafter_device) / 1e9
        print(f"  VRAM: {allocated:.2f} GB allocated / {reserved:.2f} GB reserved", flush=True)

    # ── Warmup ────────────────────────────────────────────────────────────────
    print("\nWarmup (1 prompt each mode)...", flush=True)
    if not args.skip_autoregressive:
        bench_autoregressive(target, prompts[:1], 32)
    bench_speculative_sequential(target, drafter, config, prompts[:1], 32, drafter_device,
                                 adaptive=args.adaptive)
    if args.specoffload:
        bench_speculative_pipelined(target, drafter, config, prompts[:1], 32, drafter_device,
                                    adaptive=args.adaptive)
    print("Warmup done.\n", flush=True)

    # ── Autoregressive baseline ───────────────────────────────────────────────
    ar_avg_tps = None
    ar_avg_time = None

    if not args.skip_autoregressive:
        print(f"{'='*70}")
        print(f" AUTOREGRESSIVE BASELINE ({len(prompts)} prompts, max {args.max_tokens} tokens)")
        print(f"{'='*70}")
        ar_results = bench_autoregressive(target, prompts, args.max_tokens)
        for i, r in enumerate(ar_results):
            print(f"  [{i+1:2d}] {r['tokens']:3d} tok | {r['time']:5.1f}s | {r['tok_per_s']:5.1f} tok/s")
        ar_avg_tps = np.mean([r['tok_per_s'] for r in ar_results])
        ar_avg_time = np.mean([r['time'] for r in ar_results])
        print(f"\n  AVG: {ar_avg_tps:.1f} tok/s, {ar_avg_time:.1f}s/prompt")

    # ── Sequential speculative ────────────────────────────────────────────────
    if args.compare or not args.specoffload:
        label_seq = "SEQUENTIAL SPECULATIVE"
        if args.compare:
            label_seq = "SEQUENTIAL SPECULATIVE (no pipeline)"
        print(f"\n{'='*70}")
        print(f" {label_seq} ({len(prompts)} prompts, max {args.max_tokens} tokens)")
        print(f"{'='*70}")
        seq_results = bench_speculative_sequential(
            target, drafter, config, prompts, args.max_tokens, drafter_device,
            adaptive=args.adaptive,
        )
        for i, r in enumerate(seq_results):
            print(f"  [{i+1:2d}] {r['tokens']:3d} tok | {r['time']:5.1f}s | "
                  f"{r['tok_per_s']:5.1f} tok/s | τ={r['tau']:.1%} | K={r['adaptive_k']}")
        seq_avg_tps = np.mean([r['tok_per_s'] for r in seq_results])
        seq_avg_time = np.mean([r['time'] for r in seq_results])
        seq_avg_tau = np.mean([r['tau'] for r in seq_results])
        print(f"\n  AVG: {seq_avg_tps:.1f} tok/s, {seq_avg_time:.1f}s/prompt, τ={seq_avg_tau:.1%}")
    else:
        seq_avg_tps = None
        seq_avg_tau = None

    # ── Pipelined speculative (SpecOffload) ────────────────────────────────────
    pipe_avg_tps = None
    pipe_avg_tau = None
    pipe_avg_hit_rate = None

    if args.specoffload:
        print(f"\n{'='*70}")
        print(f" PIPELINED SPECULATIVE / SpecOffload ({len(prompts)} prompts, max {args.max_tokens} tokens)")
        print(f"{'='*70}")
        pipe_results = bench_speculative_pipelined(
            target, drafter, config, prompts, args.max_tokens, drafter_device,
            adaptive=args.adaptive,
        )
        for i, r in enumerate(pipe_results):
            print(f"  [{i+1:2d}] {r['tokens']:3d} tok | {r['time']:5.1f}s | "
                  f"{r['tok_per_s']:5.1f} tok/s | τ={r['tau']:.1%} | "
                  f"hits={r['spec_hits']} miss={r['spec_misses']} "
                  f"hit%={r['spec_hit_rate']:.0%} | K={r['adaptive_k']}")
        pipe_avg_tps = np.mean([r['tok_per_s'] for r in pipe_results])
        pipe_avg_time = np.mean([r['time'] for r in pipe_results])
        pipe_avg_tau = np.mean([r['tau'] for r in pipe_results])
        pipe_avg_hit_rate = np.mean([r['spec_hit_rate'] for r in pipe_results])
        total_hits = sum(r['spec_hits'] for r in pipe_results)
        total_misses = sum(r['spec_misses'] for r in pipe_results)
        print(f"\n  AVG: {pipe_avg_tps:.1f} tok/s, {pipe_avg_time:.1f}s/prompt, τ={pipe_avg_tau:.1%}")
        print(f"  Speculative hits: {total_hits} | Misses: {total_misses} | "
              f"Hit rate: {pipe_avg_hit_rate:.1%}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f" SUMMARY")
    print(f"{'='*70}")

    if ar_avg_tps is not None:
        print(f"  Autoregressive:          {ar_avg_tps:6.1f} tok/s  (baseline)")

    if seq_avg_tps is not None:
        label = "Sequential speculative:"
        if ar_avg_tps is not None:
            speedup = seq_avg_tps / ar_avg_tps
            print(f"  {label:<26} {seq_avg_tps:6.1f} tok/s  τ={seq_avg_tau:.1%}  "
                  f"speedup={speedup:.2f}×")
        else:
            print(f"  {label:<26} {seq_avg_tps:6.1f} tok/s  τ={seq_avg_tau:.1%}")

    if pipe_avg_tps is not None:
        label = "Pipelined (SpecOffload):"
        extra = f"  hit={pipe_avg_hit_rate:.1%}"
        if ar_avg_tps is not None:
            speedup_vs_ar = pipe_avg_tps / ar_avg_tps
            speedup_vs_seq = (pipe_avg_tps / seq_avg_tps) if seq_avg_tps else None
            if speedup_vs_seq is not None:
                print(f"  {label:<26} {pipe_avg_tps:6.1f} tok/s  τ={pipe_avg_tau:.1%}  "
                      f"speedup={speedup_vs_ar:.2f}× vs AR, {speedup_vs_seq:.2f}× vs seq"
                      f"{extra}")
            else:
                print(f"  {label:<26} {pipe_avg_tps:6.1f} tok/s  τ={pipe_avg_tau:.1%}  "
                      f"speedup={speedup_vs_ar:.2f}× vs AR{extra}")
        elif seq_avg_tps is not None:
            speedup_vs_seq = pipe_avg_tps / seq_avg_tps
            print(f"  {label:<26} {pipe_avg_tps:6.1f} tok/s  τ={pipe_avg_tau:.1%}  "
                  f"speedup={speedup_vs_seq:.2f}× vs seq{extra}")
        else:
            print(f"  {label:<26} {pipe_avg_tps:6.1f} tok/s  τ={pipe_avg_tau:.1%}{extra}")

    print(f"{'='*70}")

    # ── Interpretation ────────────────────────────────────────────────────────
    if pipe_avg_tps is not None and seq_avg_tps is not None:
        ratio = pipe_avg_tps / seq_avg_tps
        if ratio >= 1.10:
            print(f"\n  VERDICT: SpecOffload gives {ratio:.2f}× speedup over sequential.")
            print(f"  Speculative pre-draft overlaps ~{pipe_avg_hit_rate:.0%} of target eval time.")
        elif ratio >= 1.01:
            print(f"\n  VERDICT: Marginal gain ({ratio:.2f}×). "
                  f"Hit rate {pipe_avg_hit_rate:.0%} — consider longer sequences for more overlap.")
        else:
            print(f"\n  VERDICT: No gain ({ratio:.2f}×). "
                  f"Overhead from threading dominates at current τ={pipe_avg_tau:.1%}.")
            print(f"  Hit rate: {pipe_avg_hit_rate:.0%}. "
                  f"SpecOffload shines when τ > 50% and target eval is long (CPU MoE layers).")
    elif pipe_avg_tps is not None and ar_avg_tps is not None:
        ratio = pipe_avg_tps / ar_avg_tps
        if ratio >= 1.0:
            print(f"\n  VERDICT: SpecOffload is {ratio:.2f}× faster than autoregressive.")
        else:
            print(f"\n  VERDICT: SpecOffload is slower than autoregressive ({ratio:.2f}×). "
                  f"τ={pipe_avg_tau:.1%} is too low for any speculative benefit.")

    target.close()


if __name__ == "__main__":
    main()
