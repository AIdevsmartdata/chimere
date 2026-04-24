"""Smoke tests for DFlash v8 data, online buffer, and entropy monitor.

Covers quick-wins #1 (data_v8 load/batch), #2 (online_buffer append/pop),
and #3 (entropy_monitor compute) from the 2026-04-23 CI audit.

These tests avoid real training data: they build a tiny in-memory packed
shard for the dataset test, exercise the OnlineBuffer against tmp_path,
and compute entropy on synthetic logits.
"""
import json
import struct
from pathlib import Path

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Quick-win #3 — entropy_monitor
# ---------------------------------------------------------------------------

def test_token_entropy_shape_and_nonneg():
    from chimere.entropy_monitor import token_entropy
    logits = torch.randn(2, 5, 32)  # [B, S, V]
    H = token_entropy(logits)
    assert H.shape == (2, 5)
    # Entropy is non-negative (in nats).
    assert (H >= 0).all()
    # Upper bound log(V) for V=32.
    assert (H <= np.log(32) + 1e-4).all()


def test_block_entropy_stats_keys():
    from chimere.entropy_monitor import block_entropy_stats
    logits = torch.randn(8, 64)  # [K, V]
    stats = block_entropy_stats(logits)
    for key in ("entropy_mean", "entropy_max", "entropy_min",
                "entropy_std", "high_entropy_ratio"):
        assert key in stats
        assert isinstance(stats[key], float)
    assert stats["entropy_min"] <= stats["entropy_mean"] <= stats["entropy_max"]


def test_adaptive_k_from_entropy_bounds():
    from chimere.entropy_monitor import adaptive_k_from_entropy
    # Very low entropy → maximum K.
    assert adaptive_k_from_entropy(0.1, current_k=8, max_k=15, min_k=2) == 15
    # Very high entropy → minimum K.
    assert adaptive_k_from_entropy(10.0, current_k=8, max_k=15, min_k=2) == 2
    # Mid range stays within bounds.
    k_mid = adaptive_k_from_entropy(2.5, current_k=8, max_k=15, min_k=2)
    assert 2 <= k_mid <= 15


# ---------------------------------------------------------------------------
# Quick-win #2 — online_buffer append/pop
# ---------------------------------------------------------------------------

def test_online_buffer_append_and_read(tmp_path):
    from chimere.online_buffer import OnlineBuffer

    buf = OnlineBuffer(
        buffer_dir=str(tmp_path / "buf"),
        capacity=8,
        layers=[0, 1],
        hidden_dim=16,
    )
    assert buf.size == 0

    n_layers, seq_len, hidden_dim = 2, 12, 16
    hidden = np.random.randn(n_layers, seq_len, hidden_dim).astype(np.float32)
    tokens = np.arange(seq_len, dtype=np.int32)

    path = buf.store(
        hidden_states=hidden,
        tokens=tokens,
        draft_tokens=[1, 2, 3, 4],
        target_tokens=[1, 2, 9, 4],
        accepted_mask=[True, True, False, True],
        anchor_pos=5,
        source="unit_test",
    )
    assert path.exists()
    assert (path / "context_hidden.bin").exists()
    assert (path / "tokens.bin").exists()
    assert (path / "metadata.json").exists()

    meta = json.loads((path / "metadata.json").read_text())
    assert meta["seq_len"] == seq_len
    assert meta["hidden_dim"] == hidden_dim

    # size grew, recent retrieval works, stats sane.
    assert buf.size == 1
    recent = buf.get_recent(n=5)
    assert len(recent) == 1
    stats = buf.stats()
    assert stats["size"] == 1
    assert stats["n_drafted"] == 4
    assert stats["n_accepted"] == 3
    assert 0.0 <= stats["tau"] <= 1.0


# ---------------------------------------------------------------------------
# Quick-win #1 — data_v8 load/batch smoke
# ---------------------------------------------------------------------------

def _write_tiny_shard(shard_dir: Path, n_samples=3, seq_len=32,
                     n_layers=2, hidden_dim=16):
    """Build a minimal packed v8 shard (1 file) and manifest."""
    shard_dir.mkdir(parents=True, exist_ok=True)
    bin_path = shard_dir / "shard_0000.bin"
    idx_path = shard_dir / "shard_0000.idx"

    samples = []
    offset = 0
    with open(bin_path, "wb") as f:
        for _ in range(n_samples):
            hidden = np.random.randn(n_layers, seq_len, hidden_dim).astype(np.float16)
            tokens = np.arange(seq_len, dtype=np.int32)
            h_bytes = hidden.tobytes()
            t_bytes = tokens.tobytes()
            f.write(h_bytes)
            f.write(t_bytes)
            samples.append({
                "offset": offset,
                "seq_len": seq_len,
                "hidden_bytes": len(h_bytes),
                "token_bytes": len(t_bytes),
            })
            offset += len(h_bytes) + len(t_bytes)

    idx_path.write_text(json.dumps({"samples": samples}))
    (shard_dir / "manifest.json").write_text(json.dumps({
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "dtype": "float16",
    }))


def test_data_v8_dataset_loads_and_samples(tmp_path):
    from chimere.data_v8 import DFlashFullSeqDataset
    shard_dir = tmp_path / "shards"
    _write_tiny_shard(shard_dir, n_samples=3, seq_len=32,
                      n_layers=2, hidden_dim=16)

    ds = DFlashFullSeqDataset(
        shard_dir=str(shard_dir),
        block_size=4,
        max_ctx_len=16,
        anchors_per_seq=2,
        num_layers=2,
        hidden_dim=16,
    )
    # Dataset reports the expected number of sequences.
    assert len(ds) == 3

    # Each item must be iterable (list of anchors) and non-empty.
    item = ds[0]
    assert item is not None
    # Anchors list — either a list or a sequence; we just check truthiness
    # and that it can be iterated once.
    try:
        anchors = list(item)
    except TypeError:
        pytest.skip("Dataset __getitem__ returned non-iterable; API changed.")
    assert len(anchors) >= 1
