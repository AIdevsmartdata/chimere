#!/usr/bin/env python3
"""Tests for chimere/entropy_monitor.py."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chimere.entropy_monitor import (
    EnergyVerifierConfig,
    EnergyVerifier,
    EntropyTracker,
    adaptive_k_from_entropy,
    block_entropy_stats,
    token_entropy,
)


def test_token_entropy_uniform():
    # Uniform distribution → max entropy = log(V)
    V = 100
    logits = torch.zeros(1, 10, V)
    H = token_entropy(logits)
    expected = torch.log(torch.tensor(V, dtype=torch.float))
    assert torch.allclose(H, expected.expand_as(H), atol=1e-4), f"Expected {expected.item():.4f}, got {H.mean().item():.4f}"
    print(f"  uniform entropy: OK (H={H.mean().item():.4f}, expected={expected.item():.4f})")


def test_token_entropy_peaked():
    # One-hot distribution → entropy ≈ 0
    V = 100
    logits = torch.full((1, 5, V), -100.0)
    logits[:, :, 0] = 100.0
    H = token_entropy(logits)
    assert H.max().item() < 0.01, f"Expected near-zero entropy, got {H.max().item():.4f}"
    print(f"  peaked entropy: OK (H={H.max().item():.6f})")


def test_token_entropy_top_k():
    V = 1000
    logits = torch.randn(2, 8, V)
    H_full = token_entropy(logits, top_k=0)
    H_topk = token_entropy(logits, top_k=64)
    # Top-k entropy should be <= full entropy (fewer terms)
    assert (H_topk <= H_full + 0.1).all(), "Top-k entropy should be <= full"
    print(f"  top-k entropy: OK (full={H_full.mean():.3f}, top64={H_topk.mean():.3f})")


def test_block_entropy_stats():
    logits = torch.randn(16, 1000)
    stats = block_entropy_stats(logits)
    assert "entropy_mean" in stats
    assert "entropy_max" in stats
    assert "entropy_std" in stats
    assert "high_entropy_ratio" in stats
    assert stats["entropy_mean"] > 0
    print(f"  block stats: OK ({stats})")


def test_adaptive_k():
    assert adaptive_k_from_entropy(0.5, 8, max_k=15, min_k=2) == 15  # low entropy → max K
    assert adaptive_k_from_entropy(6.0, 8, max_k=15, min_k=2) == 2   # high entropy → min K
    k_mid = adaptive_k_from_entropy(2.75, 8, max_k=15, min_k=2)
    assert 2 <= k_mid <= 15
    print(f"  adaptive K: OK (low→15, mid→{k_mid}, high→2)")


def test_entropy_tracker():
    tracker = EntropyTracker(window_size=5)
    for i in range(10):
        stats = {"entropy_mean": float(i), "entropy_max": float(i + 1)}
        tracker.record(stats, acceptance_rate=1.0 - i * 0.1)

    assert tracker.recent_mean_entropy > 0
    corr = tracker.entropy_tau_correlation
    assert corr is not None
    assert corr < 0, f"Expected negative correlation, got {corr:.4f}"
    print(f"  tracker: OK (mean_H={tracker.recent_mean_entropy:.2f}, corr={corr:.4f})")


def test_entropy_tracker_summary():
    tracker = EntropyTracker()
    for i in range(3):
        tracker.record({"entropy_mean": float(i)}, acceptance_rate=0.5)
    summary = tracker.summary()
    assert "recent_mean_entropy" in summary
    assert "n_blocks" in summary
    print(f"  tracker summary: OK ({summary})")


def test_entropy_tracker_escalate():
    tracker = EntropyTracker(window_size=3)
    for _ in range(5):
        tracker.record({"entropy_mean": 6.0})
    assert tracker.should_escalate(threshold=5.0)
    tracker2 = EntropyTracker(window_size=3)
    for _ in range(5):
        tracker2.record({"entropy_mean": 1.0})
    assert not tracker2.should_escalate(threshold=5.0)
    print("  tracker escalate: OK")


def test_energy_verifier_shapes():
    config = EnergyVerifierConfig(hidden_size=64, intermediate_size=32)
    verifier = EnergyVerifier(config)
    hidden = torch.randn(4, 16, 64)
    energy = verifier(hidden)
    assert energy.shape == (4,), f"Expected (4,), got {energy.shape}"
    print(f"  verifier shapes: OK (energy={energy.shape})")


def test_energy_verifier_gradient():
    config = EnergyVerifierConfig(hidden_size=64, intermediate_size=32)
    verifier = EnergyVerifier(config)
    hidden = torch.randn(4, 16, 64, requires_grad=True)
    energy = verifier(hidden)
    energy.sum().backward()
    assert hidden.grad is not None
    print("  verifier gradient: OK")


def test_energy_verifier_training_loss():
    config = EnergyVerifierConfig(hidden_size=64, intermediate_size=32)
    verifier = EnergyVerifier(config)
    hidden = torch.randn(8, 16, 64)
    accepted = torch.tensor([True, True, True, True, False, False, False, False])
    loss = verifier.training_loss(hidden, accepted)
    assert loss.dim() == 0  # scalar
    assert loss.item() > 0
    loss.backward()
    # Check gradients exist
    grad_count = sum(1 for p in verifier.parameters() if p.grad is not None)
    assert grad_count > 0
    print(f"  verifier training loss: OK (loss={loss.item():.4f})")


def test_energy_verifier_escalation():
    config = EnergyVerifierConfig(hidden_size=64, intermediate_size=32, energy_escalate=0.5)
    verifier = EnergyVerifier(config)
    hidden = torch.randn(1, 16, 64)
    result = verifier.should_escalate(hidden)
    assert isinstance(result, bool)
    print(f"  verifier escalation: OK (escalate={result})")


def test_energy_verifier_param_count():
    config = EnergyVerifierConfig(hidden_size=2048, intermediate_size=1024)
    verifier = EnergyVerifier(config)
    n_params = sum(p.numel() for p in verifier.parameters())
    print(f"  verifier params: {n_params:,} ({n_params / 1e6:.1f}M)")
    assert n_params < 10_000_000, f"Should be small, got {n_params:,}"


if __name__ == "__main__":
    print("Testing chimere/entropy_monitor.py...")
    test_token_entropy_uniform()
    test_token_entropy_peaked()
    test_token_entropy_top_k()
    test_block_entropy_stats()
    test_adaptive_k()
    test_entropy_tracker()
    test_entropy_tracker_summary()
    test_entropy_tracker_escalate()
    test_energy_verifier_shapes()
    test_energy_verifier_gradient()
    test_energy_verifier_training_loss()
    test_energy_verifier_escalation()
    test_energy_verifier_param_count()
    print("\nAll entropy monitor tests passed!")
