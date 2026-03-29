#!/usr/bin/env python3
"""Tests for chimere/engram.py — EngramModule, NgramHasher, ContextGating."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chimere.engram import EngramConfig, EngramModule, NgramHasher, EngramTable, ContextGating


def test_ngram_hasher_shapes():
    config = EngramConfig(table_size=1024, num_tables=4, ngram_sizes=[2, 3])
    hasher = NgramHasher(config)
    tokens = torch.randint(0, 1000, (2, 32))
    indices = hasher(tokens)
    expected_k = config.num_tables * len(config.ngram_sizes)  # 4 * 2 = 8
    assert indices.shape == (2, 32, expected_k), f"Expected (2, 32, {expected_k}), got {indices.shape}"
    assert (indices >= 0).all() and (indices < config.table_size).all()
    print(f"  hasher shapes: OK ({indices.shape})")


def test_ngram_hasher_deterministic():
    config = EngramConfig(table_size=1024, num_tables=4)
    hasher = NgramHasher(config)
    tokens = torch.randint(0, 1000, (1, 16))
    idx1 = hasher(tokens)
    idx2 = hasher(tokens)
    assert torch.equal(idx1, idx2), "Hasher should be deterministic"
    print("  hasher deterministic: OK")


def test_ngram_hasher_different_ngrams():
    config = EngramConfig(table_size=1024, num_tables=4)
    hasher = NgramHasher(config)
    tokens_a = torch.tensor([[10, 20, 30, 40]])
    tokens_b = torch.tensor([[10, 20, 30, 50]])
    idx_a = hasher(tokens_a)
    idx_b = hasher(tokens_b)
    # Last position should differ (different trigram context)
    assert not torch.equal(idx_a[:, -1], idx_b[:, -1]), "Different n-grams should hash differently"
    print("  hasher different n-grams: OK")


def test_engram_table_shapes():
    config = EngramConfig(table_size=256, num_tables=4, hidden_size=64, ngram_sizes=[2, 3])
    table = EngramTable(config)
    K = config.num_tables * len(config.ngram_sizes)
    indices = torch.randint(0, config.table_size, (2, 16, K))
    engrams = table(indices)
    assert engrams.shape == (2, 16, K, 64), f"Expected (2, 16, {K}, 64), got {engrams.shape}"
    print(f"  table shapes: OK ({engrams.shape})")


def test_engram_table_small_init():
    config = EngramConfig(table_size=256, num_tables=4, hidden_size=64)
    table = EngramTable(config)
    # Weights should be initialized small (std=0.01)
    std = table.table.weight.std().item()
    assert std < 0.05, f"Expected small init, got std={std:.4f}"
    print(f"  table init: OK (std={std:.4f})")


def test_context_gating_shapes():
    config = EngramConfig(hidden_size=64, num_tables=4, ngram_sizes=[2, 3])
    gating = ContextGating(config)
    K = config.num_tables * len(config.ngram_sizes)
    hidden = torch.randn(2, 16, 64)
    engrams = torch.randn(2, 16, K, 64)
    contribution, alphas = gating(hidden, engrams)
    assert contribution.shape == (2, 16, 64)
    assert alphas.shape == (2, 16, K)
    print(f"  gating shapes: OK (alphas: {alphas.shape})")


def test_context_gating_bias():
    config = EngramConfig(hidden_size=64, gate_bias=-5.0)
    gating = ContextGating(config)
    K = config.num_tables * len(config.ngram_sizes)
    hidden = torch.randn(1, 8, 64)
    engrams = torch.randn(1, 8, K, 64) * 0.01  # small engrams
    _, alphas = gating(hidden, engrams)
    # With bias=-5.0 and small engrams, alphas should be near sigmoid(-5) ≈ 0.007
    mean_alpha = alphas.mean().item()
    assert mean_alpha < 0.1, f"Expected low alpha with negative bias, got {mean_alpha:.4f}"
    print(f"  gating bias: OK (mean_alpha={mean_alpha:.4f})")


def test_engram_module_forward():
    config = EngramConfig(hidden_size=64, table_size=256, num_tables=4, vocab_size=1000)
    module = EngramModule(config)
    tokens = torch.randint(0, 1000, (2, 32))
    hidden = torch.randn(2, 32, 64)
    enriched = module(tokens, hidden)
    assert enriched.shape == hidden.shape
    # Should be close to hidden (small init + negative gate bias)
    diff = (enriched - hidden).norm() / hidden.norm()
    assert diff < 0.5, f"Enriched should be close to hidden with small init, diff={diff:.4f}"
    print(f"  module forward: OK (relative diff={diff:.4f})")


def test_engram_module_diagnostics():
    config = EngramConfig(hidden_size=64, table_size=256, num_tables=4, vocab_size=1000)
    module = EngramModule(config)
    tokens = torch.randint(0, 1000, (1, 16))
    hidden = torch.randn(1, 16, 64)
    enriched, diag = module(tokens, hidden, return_diagnostics=True)
    assert "mean_alpha" in diag
    assert "active_ratio" in diag
    assert "contribution_norm" in diag
    print(f"  module diagnostics: OK ({diag})")


def test_engram_module_gradient():
    config = EngramConfig(hidden_size=64, table_size=256, num_tables=4, vocab_size=1000)
    module = EngramModule(config)
    tokens = torch.randint(0, 1000, (1, 8))
    hidden = torch.randn(1, 8, 64, requires_grad=True)
    enriched = module(tokens, hidden)
    loss = enriched.sum()
    loss.backward()
    assert hidden.grad is not None
    # Check that table weights received gradients
    assert module.table.table.weight.grad is not None
    print("  module gradient: OK")


def test_engram_update():
    config = EngramConfig(hidden_size=64, table_size=256, num_tables=4, vocab_size=1000)
    module = EngramModule(config)
    tokens = torch.randint(0, 1000, (1, 8))
    target = torch.randn(1, 8, 64)

    # Record weights before
    indices = module.hasher(tokens)
    offsets = torch.arange(indices.shape[-1]) * config.table_size
    flat_idx = (indices + offsets).reshape(-1)
    before = module.table.table.weight[flat_idx].clone()

    module.update_engrams(tokens, target, lr=0.5)

    after = module.table.table.weight[flat_idx]
    assert not torch.equal(before, after), "Weights should have changed"
    print("  module update: OK")


def test_engram_cpu_gpu_split():
    if not torch.cuda.is_available():
        print("  cpu/gpu split: SKIPPED (no CUDA)")
        return
    config = EngramConfig(hidden_size=64, table_size=256, num_tables=4, vocab_size=1000, device="cpu")
    module = EngramModule(config)
    tokens = torch.randint(0, 1000, (1, 8)).cuda()
    hidden = torch.randn(1, 8, 64).cuda()
    enriched = module(tokens, hidden)
    assert enriched.device.type == "cuda"
    assert module.table.table.weight.device.type == "cpu"
    print("  cpu/gpu split: OK (table=CPU, compute=CUDA)")


if __name__ == "__main__":
    print("Testing chimere/engram.py...")
    test_ngram_hasher_shapes()
    test_ngram_hasher_deterministic()
    test_ngram_hasher_different_ngrams()
    test_engram_table_shapes()
    test_engram_table_small_init()
    test_context_gating_shapes()
    test_context_gating_bias()
    test_engram_module_forward()
    test_engram_module_diagnostics()
    test_engram_module_gradient()
    test_engram_update()
    test_engram_cpu_gpu_split()
    print("\nAll engram tests passed!")
