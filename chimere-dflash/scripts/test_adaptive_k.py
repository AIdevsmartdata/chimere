#!/usr/bin/env python3
"""
Test adaptive K selection logic for DFlash speculative decoding.

Pure Python, no model loading required. Tests:
  1. Adaptive K selection given recent_tau history
  2. Draft truncation correctness (truncate block_size=15 → K, verify K, accept/reject)

All tests use assert statements and print PASS/FAIL.

Background:
  block_size=16 means the drafter always generates 16 tokens internally.
  Position 0 is the anchor (already-verified token from target).
  Positions 1..15 are the speculative draft tokens (K_max = 15 draftable positions).

  Adaptive K: based on recent acceptance rate τ, select how many of those 15
  tokens to actually submit for verification (K ∈ {1..15}). This avoids wasting
  a target eval call on tokens the drafter is likely to get wrong.

  Verification protocol (from benchmark_wallclock.py bench_speculative):
    target_preds = [last_target_pred] + incr_argmax[:-1]
    for j in range(len(draft_tokens)):
        if draft_tokens[j] == target_preds[j]:
            n_accepted += 1
        else:
            break
    if n_accepted < len(draft_tokens):
        # partial: restore state, apply correction = target_preds[n_accepted]
    else:
        # all accepted: bonus = incr_argmax[-1]
"""

import sys

# ──────────────────────────────────────────────────────────────────────────────
# Reference implementation of adaptive K selection
# ──────────────────────────────────────────────────────────────────────────────

K_MAX = 15          # block_size=16, position 0 = anchor, so 15 draftable slots
K_MIN = 1           # always draft at least 1 token
HISTORY_LEN = 8     # rolling window for τ estimation
TAU_HIGH = 0.40     # τ above this → use full K_MAX
TAU_LOW  = 0.10     # τ below this → fall back to K_MIN


def select_k(recent_tau: list[float]) -> int:
    """
    Select how many draft tokens to submit for verification.

    Strategy:
      - Empty history          → K_MAX (optimistic start)
      - τ_avg >= TAU_HIGH      → K_MAX  (drafter is good, maximize throughput)
      - TAU_LOW <= τ_avg < TAU_HIGH → linear interpolation in [K_MIN, K_MAX]
      - τ_avg < TAU_LOW        → K_MIN  (drafter is poor, minimize wasted verifications)

    Args:
        recent_tau: list of per-block acceptance rates (floats in [0, 1]).
                    At most HISTORY_LEN last values are used.

    Returns:
        K (int) — number of draft tokens to verify, in [K_MIN, K_MAX].
    """
    if not recent_tau:
        return K_MAX

    window = recent_tau[-HISTORY_LEN:]
    tau_avg = sum(window) / len(window)

    if tau_avg >= TAU_HIGH:
        return K_MAX
    elif tau_avg < TAU_LOW:
        return K_MIN
    else:
        # Linear interpolation: τ_avg in [TAU_LOW, TAU_HIGH] → K in [K_MIN, K_MAX]
        frac = (tau_avg - TAU_LOW) / (TAU_HIGH - TAU_LOW)  # 0.0 → 1.0
        k_float = K_MIN + frac * (K_MAX - K_MIN)
        return max(K_MIN, min(K_MAX, round(k_float)))


# ──────────────────────────────────────────────────────────────────────────────
# Reference implementation of the verification logic (pure Python simulation)
# ──────────────────────────────────────────────────────────────────────────────

def verify_draft(draft_tokens: list[int], target_preds: list[int]) -> tuple[int, int, list[int]]:
    """
    Simulate the speculative decoding verification step.

    Mirrors bench_speculative() from benchmark_wallclock.py:
        target_preds = [last_target_pred] + incr_argmax[:-1]
        n_accepted = 0
        for j in range(len(draft_tokens)):
            if draft_tokens[j] == target_preds[j]:
                n_accepted += 1
            else:
                break

    Args:
        draft_tokens:  list of K token IDs proposed by the drafter.
        target_preds:  list of K token IDs the target would have chosen.
                       target_preds[j] is what the target predicts at draft position j.
                       Concretely: [last_target_pred_before_block] + incr_argmax[0..K-2]

    Returns:
        (n_accepted, correction_or_bonus, accepted_tokens)
          - n_accepted: number of sequentially accepted tokens
          - correction_or_bonus: target_preds[n_accepted] if partial,
                                 or incr_argmax[-1] (bonus) if all accepted
          - accepted_tokens: list of tokens added to output
    """
    assert len(draft_tokens) == len(target_preds), (
        f"len mismatch: draft={len(draft_tokens)}, preds={len(target_preds)}"
    )
    K = len(draft_tokens)

    n_accepted = 0
    for j in range(K):
        if draft_tokens[j] == target_preds[j]:
            n_accepted += 1
        else:
            break

    accepted_tokens = list(draft_tokens[:n_accepted])

    if n_accepted < K:
        # Partial accept: target's correction at first mismatch position
        correction = target_preds[n_accepted]
        accepted_tokens.append(correction)
        return n_accepted, correction, accepted_tokens
    else:
        # All K accepted: bonus token (incr_argmax[-1])
        # In real impl: bonus = int(incr_argmax[-1]), separate eval_incr([bonus])
        # Here we signal "all accepted" with sentinel None for bonus (caller supplies it)
        return n_accepted, None, accepted_tokens


def compute_tau_for_block(n_accepted: int, K: int) -> float:
    """τ for a single block = accepted / K (draftable positions)."""
    return n_accepted / K if K > 0 else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Test helpers
# ──────────────────────────────────────────────────────────────────────────────

_tests_run   = 0
_tests_passed = 0
_tests_failed = 0


def run_test(name: str, fn) -> bool:
    global _tests_run, _tests_passed, _tests_failed
    _tests_run += 1
    try:
        fn()
        print(f"  PASS  {name}")
        _tests_passed += 1
        return True
    except AssertionError as e:
        msg = str(e) if str(e) else "(no message)"
        print(f"  FAIL  {name}")
        print(f"        AssertionError: {msg}")
        _tests_failed += 1
        return False
    except Exception as e:
        print(f"  FAIL  {name}")
        print(f"        Exception ({type(e).__name__}): {e}")
        _tests_failed += 1
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Section 1: Adaptive K selection tests
# ──────────────────────────────────────────────────────────────────────────────

def test_empty_history_returns_k_max():
    k = select_k([])
    assert k == K_MAX, f"empty history: expected K_MAX={K_MAX}, got {k}"


def test_all_high_tau_returns_k_max():
    # τ = 1.0 everywhere → use full block
    history = [1.0] * 8
    k = select_k(history)
    assert k == K_MAX, f"all-high τ: expected K_MAX={K_MAX}, got {k}"


def test_all_high_tau_at_threshold_returns_k_max():
    # Exactly at TAU_HIGH
    history = [TAU_HIGH] * 5
    k = select_k(history)
    assert k == K_MAX, f"τ=TAU_HIGH: expected K_MAX={K_MAX}, got {k}"


def test_all_low_tau_returns_k_min():
    history = [0.0] * 8
    k = select_k(history)
    assert k == K_MIN, f"all-zero τ: expected K_MIN={K_MIN}, got {k}"


def test_all_low_tau_at_threshold_returns_k_min():
    # Just below TAU_LOW
    history = [TAU_LOW - 0.001] * 8
    k = select_k(history)
    assert k == K_MIN, f"τ just below TAU_LOW: expected K_MIN={K_MIN}, got {k}"


def test_mid_tau_gives_intermediate_k():
    # τ exactly halfway between TAU_LOW and TAU_HIGH
    tau_mid = (TAU_LOW + TAU_HIGH) / 2.0  # 0.25
    history = [tau_mid] * 8
    k = select_k(history)
    # Expected: round(K_MIN + 0.5 * (K_MAX - K_MIN)) = round(1 + 0.5*14) = round(8) = 8
    k_expected = round(K_MIN + 0.5 * (K_MAX - K_MIN))
    assert k == k_expected, (
        f"τ_mid={tau_mid:.3f}: expected K={k_expected}, got {k}"
    )


def test_k_always_in_valid_range():
    import random
    rng = random.Random(42)
    for _ in range(200):
        n = rng.randint(0, 20)
        history = [rng.random() for _ in range(n)]
        k = select_k(history)
        assert K_MIN <= k <= K_MAX, (
            f"K={k} out of range [{K_MIN}, {K_MAX}] for history={history}"
        )


def test_only_last_history_len_entries_used():
    # Pad with old high-τ values, then add new low-τ values
    old_high = [1.0] * 100
    new_low   = [0.0] * HISTORY_LEN
    history   = old_high + new_low
    k = select_k(history)
    # Only the last HISTORY_LEN entries (all 0.0) should matter → K_MIN
    assert k == K_MIN, (
        f"should use only last {HISTORY_LEN} entries → K_MIN={K_MIN}, got {k}"
    )


def test_transition_high_to_low():
    # Start with good τ, drafter degrades
    history_good = [0.8, 0.7, 0.6, 0.5]
    k_before = select_k(history_good)

    history_bad = history_good + [0.0, 0.0, 0.0, 0.0]
    k_after = select_k(history_bad)

    assert k_before > k_after, (
        f"K should decrease as τ degrades: before={k_before}, after={k_after}"
    )


def test_transition_low_to_high():
    # Start with bad τ, drafter improves
    history_bad = [0.0, 0.0, 0.0, 0.0]
    k_before = select_k(history_bad)

    history_good = history_bad + [1.0, 1.0, 1.0, 1.0]
    k_after = select_k(history_good)

    assert k_after > k_before, (
        f"K should increase as τ improves: before={k_before}, after={k_after}"
    )


def test_single_entry_history():
    # One entry at exactly TAU_HIGH → K_MAX
    assert select_k([TAU_HIGH]) == K_MAX
    # One entry at 0.0 → K_MIN
    assert select_k([0.0]) == K_MIN


def test_k_monotone_with_tau():
    """Higher τ_avg should never produce a smaller K than lower τ_avg."""
    tau_levels = [i / 20.0 for i in range(21)]  # 0.0, 0.05, 0.10, ... 1.0
    k_values = [select_k([t] * HISTORY_LEN) for t in tau_levels]

    for i in range(len(k_values) - 1):
        assert k_values[i] <= k_values[i + 1], (
            f"K not monotone: τ={tau_levels[i]:.2f}→K={k_values[i]}, "
            f"τ={tau_levels[i+1]:.2f}→K={k_values[i+1]}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Section 2: Draft truncation correctness tests
# ──────────────────────────────────────────────────────────────────────────────
# The drafter always generates block_size=16 tokens internally (pos 0 = anchor,
# pos 1..15 = draft). Adaptive K truncates the draft before sending to target.
#
# Truncation rule:
#   full_draft = draft_ids[1:16]   # positions 1..15 from the block (15 tokens)
#   truncated  = full_draft[:K]    # only submit K of them
#   verified   = target.eval_incr(truncated)  → incr_argmax[0..K-1]
#   target_preds = [last_target_pred] + incr_argmax[:-1]  # length K
# ──────────────────────────────────────────────────────────────────────────────

def _make_target_preds(last_pred: int, incr_argmax: list[int]) -> list[int]:
    """
    Build target_preds as done in bench_speculative:
        target_preds = [last_target_pred] + incr_argmax[:-1]
    This has length K (same as draft_tokens submitted).
    """
    return [last_pred] + list(incr_argmax[:-1])


def test_truncate_15_to_4_all_accepted():
    """Truncate 15-token draft to K=4, verify 4, all accepted."""
    # Simulate: block_size=16, anchor at pos 0
    # drafter generates positions 1..15: tokens 101..115
    full_draft_15 = list(range(101, 116))  # [101, 102, ..., 115]

    K = 4
    draft_k = full_draft_15[:K]  # [101, 102, 103, 104]
    assert len(draft_k) == K

    # Target prediction at position 0 = last_target_pred (the anchor itself,
    # which is what the target predicted before this block)
    last_target_pred = 101  # target predicted 101 at anchor position

    # eval_incr(draft_k) → incr_argmax: target predicts what comes AFTER each input
    # If all draft are accepted, then incr_argmax[j] == draft_k[j+1] for j in 0..K-2
    # and incr_argmax[K-1] is the bonus prediction
    incr_argmax = [102, 103, 104, 999]  # 102=after 101, 103=after 102, 104=after 103, bonus=999

    target_preds = _make_target_preds(last_target_pred, incr_argmax)
    # = [101, 102, 103, 104]
    assert target_preds == [101, 102, 103, 104], f"target_preds={target_preds}"

    n_accepted, correction_or_bonus, accepted_tokens = verify_draft(draft_k, target_preds)

    assert n_accepted == K, f"expected all {K} accepted, got {n_accepted}"
    assert correction_or_bonus is None, "all accepted → bonus (None sentinel)"
    assert accepted_tokens == [101, 102, 103, 104], f"accepted_tokens={accepted_tokens}"


def test_truncate_15_to_4_first_rejected():
    """Truncate to K=4, verify 4, first token rejected immediately."""
    full_draft_15 = list(range(201, 216))
    K = 4
    draft_k = full_draft_15[:K]  # [201, 202, 203, 204]

    last_target_pred = 999  # target predicted 999, drafter predicted 201 → mismatch at j=0
    incr_argmax = [202, 203, 204, 205]  # what target says after each draft token

    target_preds = _make_target_preds(last_target_pred, incr_argmax)
    # = [999, 202, 203, 204]

    n_accepted, correction, accepted_tokens = verify_draft(draft_k, target_preds)

    assert n_accepted == 0, f"expected 0 accepted (first mismatch), got {n_accepted}"
    assert correction == 999, f"correction should be target_preds[0]=999, got {correction}"
    # accepted_tokens = [] (0 draft accepted) + [correction=999]
    assert accepted_tokens == [999], f"accepted_tokens={accepted_tokens}"


def test_truncate_15_to_4_partial_accept_at_2():
    """Truncate to K=4, verify 4, accept 2 then reject."""
    full_draft_15 = list(range(301, 316))
    K = 4
    draft_k = full_draft_15[:K]  # [301, 302, 303, 304]

    # Target preds match for j=0,1 but diverge at j=2
    last_target_pred = 301  # match j=0
    # incr_argmax: [302, 303, WRONG, ...]
    # target_preds = [301, 302, 303, WRONG]  → match at j=0,1,2, mismatch at j=3? Let's be precise.
    # draft_k[0]=301 vs target_preds[0]=301 → match (n_accepted++)
    # draft_k[1]=302 vs target_preds[1]=302 → match (n_accepted++)
    # draft_k[2]=303 vs target_preds[2]=888 → MISMATCH → stop
    incr_argmax = [302, 888, 304, 305]
    target_preds = _make_target_preds(last_target_pred, incr_argmax)
    # = [301, 302, 888, 304]

    n_accepted, correction, accepted_tokens = verify_draft(draft_k, target_preds)

    assert n_accepted == 2, f"expected 2 accepted, got {n_accepted}"
    assert correction == 888, f"correction should be target_preds[2]=888, got {correction}"
    # accepted = draft_k[:2] = [301, 302], then correction=888
    assert accepted_tokens == [301, 302, 888], f"accepted_tokens={accepted_tokens}"


def test_truncation_does_not_affect_verification_indices():
    """
    Verify that truncating draft from 15→K doesn't misalign verification indices.

    The verification protocol only cares about the K tokens submitted.
    Tokens beyond K are simply not evaluated — their correctness is irrelevant.
    This test proves the indices stay consistent after truncation.
    """
    # Full 15-token draft from drafter: tokens 1..15
    full_draft_15 = list(range(1, 16))  # [1, 2, ..., 15]

    last_target_pred = 1  # target's prediction at anchor: token 1

    for K in [1, 4, 8, 15]:
        draft_k = full_draft_15[:K]

        # Construct a scenario where target agrees on all K tokens
        # incr_argmax: target predicts draft_k[j+1] after seeing draft_k[j]
        # For K=1, incr_argmax has 1 element (bonus only)
        if K > 1:
            incr_argmax = list(draft_k[1:]) + [999]  # length K
        else:
            incr_argmax = [999]  # only bonus

        target_preds = _make_target_preds(last_target_pred, incr_argmax)
        # Should be exactly draft_k if all agree
        # target_preds = [last_target_pred] + incr_argmax[:-1]
        # = [1] + draft_k[1:] = draft_k ✓

        assert target_preds == draft_k, (
            f"K={K}: target_preds should equal draft_k when all agree\n"
            f"  target_preds={target_preds}\n  draft_k={draft_k}"
        )

        n_accepted, sentinel, accepted = verify_draft(draft_k, target_preds)
        assert n_accepted == K, f"K={K}: all should be accepted, got {n_accepted}"
        assert sentinel is None, f"K={K}: all accepted → bonus sentinel None"
        assert accepted == list(draft_k), (
            f"K={K}: accepted_tokens should be draft_k, got {accepted}"
        )


def test_truncation_tau_accounting():
    """
    τ computed from truncated draft must use the truncated K as denominator,
    not the full 15.

    If K=4 and n_accepted=2, then τ=0.5 for this block.
    Using the full K_MAX=15 as denominator would give τ=0.133, severely
    underestimating drafter quality.
    """
    K_full = K_MAX  # 15
    K_trunc = 4
    n_accepted = 2

    tau_correct   = compute_tau_for_block(n_accepted, K_trunc)  # 2/4 = 0.50
    tau_incorrect = compute_tau_for_block(n_accepted, K_full)   # 2/15 ≈ 0.133

    assert abs(tau_correct - 0.50) < 1e-9, (
        f"τ_correct should be 0.50, got {tau_correct}"
    )
    assert tau_correct > tau_incorrect, (
        f"τ with truncated K should be higher: {tau_correct} vs {tau_incorrect}"
    )

    # Using wrong τ would push select_k toward K_MIN — verify this:
    history_correct   = [tau_correct]   * HISTORY_LEN   # τ=0.5 → should give intermediate K
    history_incorrect = [tau_incorrect] * HISTORY_LEN   # τ=0.133 → might give K_MIN

    k_correct   = select_k(history_correct)
    k_incorrect = select_k(history_incorrect)

    # Both in valid range
    assert K_MIN <= k_correct   <= K_MAX
    assert K_MIN <= k_incorrect <= K_MAX

    # Correct accounting gives higher or equal K
    assert k_correct >= k_incorrect, (
        f"correct τ accounting should give K≥K from incorrect: "
        f"k_correct={k_correct}, k_incorrect={k_incorrect}"
    )


def test_k_equals_1_edge_case():
    """K=1: submit only 1 draft token, exactly one comparison, then correction/bonus."""
    draft_k = [42]
    last_target_pred = 42  # match
    incr_argmax = [100]    # bonus
    target_preds = _make_target_preds(last_target_pred, incr_argmax)
    # = [42] + [] = [42]
    assert target_preds == [42], f"target_preds={target_preds}"

    n_accepted, sentinel, accepted = verify_draft(draft_k, target_preds)
    assert n_accepted == 1, f"K=1 all accepted: got {n_accepted}"
    assert sentinel is None
    assert accepted == [42]

    # Now mismatch
    target_preds_miss = [77]
    n_accepted, correction, accepted = verify_draft([42], target_preds_miss)
    assert n_accepted == 0
    assert correction == 77
    assert accepted == [77]


def test_verify_preserves_token_identity():
    """
    Tokens that were not submitted for verification must not appear in the output.

    Full draft has 15 tokens. If K=4 and n_accepted=2, the output is:
      draft_k[0], draft_k[1], correction
    Tokens draft_k[4..14] (the unsubmitted ones) must never appear.
    """
    full_draft_15 = list(range(500, 515))  # [500..514]
    K = 4
    draft_k = full_draft_15[:K]           # [500, 501, 502, 503]
    unsubmitted = full_draft_15[K:]        # [504..514]

    last_target_pred = 500  # j=0 matches
    # j=1: draft 501 vs target_preds[1]: let's say mismatch
    incr_argmax = [888, 502, 503, 504]
    target_preds = _make_target_preds(last_target_pred, incr_argmax)
    # = [500, 888, 502, 503]
    # j=0: 500==500 ✓, j=1: 501 vs 888 ✗ → n_accepted=1, correction=888

    n_accepted, correction, accepted = verify_draft(draft_k, target_preds)
    assert n_accepted == 1
    assert correction == 888
    assert accepted == [500, 888]

    # No unsubmitted token should appear in the accepted list
    for tok in unsubmitted:
        assert tok not in accepted, (
            f"unsubmitted token {tok} leaked into accepted={accepted}"
        )


def test_adaptive_k_feedback_loop():
    """
    Simulate 20 blocks of speculative decoding with adaptive K.

    Block outcome: for simplicity, acceptance rate alternates between
    high (0.8) and low (0.05) phases.  Verify that K tracks τ correctly.
    """
    recent_tau = []
    k_history = []

    # Phase 1: high τ (10 blocks)
    for _ in range(10):
        tau_block = 0.80
        recent_tau.append(tau_block)
        k = select_k(recent_tau)
        k_history.append(k)

    # After 10 high-τ blocks, K should be K_MAX
    final_k_phase1 = k_history[-1]
    assert final_k_phase1 == K_MAX, (
        f"after high-τ phase, expected K_MAX={K_MAX}, got {final_k_phase1}"
    )

    # Phase 2: low τ (10 blocks)
    for _ in range(10):
        tau_block = 0.02
        recent_tau.append(tau_block)
        k = select_k(recent_tau)
        k_history.append(k)

    # After 10 low-τ blocks (which fill the history window), K should be K_MIN
    final_k_phase2 = k_history[-1]
    assert final_k_phase2 == K_MIN, (
        f"after low-τ phase, expected K_MIN={K_MIN}, got {final_k_phase2}"
    )

    # Phase transition: K must have decreased from phase 1 to phase 2
    max_k_phase2 = max(k_history[10:])
    assert max_k_phase2 <= K_MAX, "K never exceeds K_MAX"
    assert final_k_phase2 < final_k_phase1, (
        f"K should decrease from phase 1 to phase 2: "
        f"{final_k_phase1} → {final_k_phase2}"
    )


def test_tokens_per_target_call_formula():
    """
    Validate the theoretical tokens/target_call formula for different K and τ.

    In speculative decoding:
      - Each target call verifies K draft tokens
      - Expected accepted = K * τ (if independent, which is an approximation)
      - Tokens produced = n_accepted + 1 (the +1 is the correction or bonus token)
      - tokens/call ≈ K * τ + 1

    This is the key metric that determines actual speedup vs autoregressive.
    For K_MAX=15, τ=0.027 (our current 75K checkpoint):
      tokens/call ≈ 15 * 0.027 + 1 = 1.4  (barely above AR baseline of 1.0)

    For adaptive K=4, τ=0.027:
      tokens/call ≈ 4 * 0.027 + 1 = 1.1  (even worse — small K hurts at low τ)

    → Conclusion: adaptive K only helps when τ is moderate (0.1–0.4),
      NOT when τ is uniformly low. At τ=0.027 we need better drafter data first.
    """
    def expected_tokens_per_call(K: int, tau: float) -> float:
        return K * tau + 1.0

    # At very low τ, larger K always gives more tokens/call
    tau_low = 0.027
    assert expected_tokens_per_call(15, tau_low) > expected_tokens_per_call(4, tau_low), (
        "At low τ, K=15 > K=4 for tokens/call — adaptive K doesn't help"
    )

    # At moderate τ, the difference is larger and K=15 dominates even more
    tau_mid = 0.30
    tpc_15 = expected_tokens_per_call(15, tau_mid)
    tpc_4  = expected_tokens_per_call(4,  tau_mid)
    assert tpc_15 > tpc_4, f"K=15 should dominate K=4 at τ={tau_mid}"
    assert tpc_15 > 5.0, f"K=15, τ=0.30 → should give >5 tok/call, got {tpc_15}"

    # At τ=0, all K values give exactly 1 token/call (only correction)
    for K in [1, 4, 8, 15]:
        tpc = expected_tokens_per_call(K, tau=0.0)
        assert abs(tpc - 1.0) < 1e-9, f"K={K}, τ=0: expected 1.0 tok/call, got {tpc}"

    # At τ=1 (perfect drafter), tokens/call = K + 1 (all accepted + bonus)
    for K in [1, 4, 8, 15]:
        tpc = expected_tokens_per_call(K, tau=1.0)
        assert abs(tpc - (K + 1)) < 1e-9, f"K={K}, τ=1: expected {K+1} tok/call, got {tpc}"


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*65}")
    print(f"  DFlash Adaptive K — Pure Logic Test Suite")
    print(f"  K_MAX={K_MAX}, K_MIN={K_MIN}, TAU_HIGH={TAU_HIGH}, TAU_LOW={TAU_LOW}")
    print(f"{'='*65}\n")

    print("─── Section 1: Adaptive K selection ───────────────────────────")
    run_test("empty history → K_MAX",                    test_empty_history_returns_k_max)
    run_test("all τ=1.0 → K_MAX",                        test_all_high_tau_returns_k_max)
    run_test("τ=TAU_HIGH exactly → K_MAX",               test_all_high_tau_at_threshold_returns_k_max)
    run_test("all τ=0.0 → K_MIN",                        test_all_low_tau_returns_k_min)
    run_test("τ just below TAU_LOW → K_MIN",             test_all_low_tau_at_threshold_returns_k_min)
    run_test("τ midpoint → intermediate K",              test_mid_tau_gives_intermediate_k)
    run_test("K always in [K_MIN, K_MAX] (random)",      test_k_always_in_valid_range)
    run_test("only last HISTORY_LEN entries used",        test_only_last_history_len_entries_used)
    run_test("K decreases as τ degrades",                 test_transition_high_to_low)
    run_test("K increases as τ recovers",                 test_transition_low_to_high)
    run_test("single-entry history edge case",            test_single_entry_history)
    run_test("K is monotone non-decreasing in τ",        test_k_monotone_with_tau)

    print()
    print("─── Section 2: Draft truncation correctness ────────────────────")
    run_test("truncate 15→4, all K accepted",             test_truncate_15_to_4_all_accepted)
    run_test("truncate 15→4, first token rejected",       test_truncate_15_to_4_first_rejected)
    run_test("truncate 15→4, partial accept at j=2",      test_truncate_15_to_4_partial_accept_at_2)
    run_test("truncation preserves verification indices", test_truncation_does_not_affect_verification_indices)
    run_test("τ accounting uses truncated K as denom",    test_truncation_tau_accounting)
    run_test("K=1 edge case (min draft)",                 test_k_equals_1_edge_case)
    run_test("unsubmitted tokens never leak to output",   test_verify_preserves_token_identity)
    run_test("adaptive K feedback loop (20 blocks)",      test_adaptive_k_feedback_loop)
    run_test("tokens/target_call formula sanity",         test_tokens_per_target_call_formula)

    print()
    print(f"{'='*65}")
    print(f"  Results: {_tests_passed}/{_tests_run} passed, {_tests_failed} failed")
    if _tests_failed == 0:
        print("  ALL TESTS PASSED")
    else:
        print(f"  {_tests_failed} TEST(S) FAILED")
    print(f"{'='*65}\n")

    return 0 if _tests_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
