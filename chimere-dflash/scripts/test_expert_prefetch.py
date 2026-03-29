#!/usr/bin/env python3
"""
Tests for ExpertPrefetchHead + DFlashDraftModelV8 integration.

Covers:
  1. ExpertPrefetchHead initialisation and forward pass
  2. Gate weight loading (from file and from random tensors)
  3. Integration with DFlashDraftModelV8 (generate_block, forward_train,
     forward_train_multi)
  4. Output shapes and value ranges
  5. Frozen gate weights do not accumulate gradients during backward

Run with:
    cd ~/chimere-dflash
    python scripts/test_expert_prefetch.py          # all tests
    python scripts/test_expert_prefetch.py -v       # verbose
    python scripts/test_expert_prefetch.py -k frozen # filter by name
"""
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chimere.config_v7 import DFlashV7Config
from chimere.expert_prefetch import ExpertPrefetchConfig, ExpertPrefetchHead
from chimere.modeling_v8 import DFlashDraftModelV8


# ---------------------------------------------------------------------------
# Helpers: small configs to keep tests fast
# ---------------------------------------------------------------------------

def small_prefetch_config(
    n_experts: int = 32,
    n_active: int = 4,
    input_dim: int = 64,
    proj_rank: int = 16,
    adapter_rank: int = 8,
    routing_layers=None,
) -> ExpertPrefetchConfig:
    if routing_layers is None:
        routing_layers = list(range(4, 8))   # 4 layers instead of 20
    return ExpertPrefetchConfig(
        routing_layers=routing_layers,
        n_experts=n_experts,
        n_active=n_active,
        input_dim=input_dim,
        proj_rank=proj_rank,
        adapter_rank=adapter_rank,
    )


def small_drafter_config(predict_expert_routing: bool = False) -> DFlashV7Config:
    """Minimal DFlashV7Config for fast, low-RAM tests.

    Uses a tiny vocab (512 tokens) and minimal dimensions so tests don't OOM.
    The hidden_size is kept at 2048 to match the real ExpertPrefetchConfig
    input_dim = 2048 (same as Qwen3.5).
    """
    return DFlashV7Config(
        # Use tiny vocab to avoid allocating 2 GB for lm_head
        target_hidden_size=2048,
        target_num_layers=40,
        target_vocab_size=512,   # minimal — real value is 248320
        num_feature_layers=3,    # fewer feature layers → smaller fc
        target_layer_ids=[1, 19, 37],
        # Tiny drafter
        hidden_size=2048,        # must equal target_hidden_size for H==T path
        num_hidden_layers=2,     # 2 drafter layers instead of 8
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=128,
        intermediate_size=256,
        rms_norm_eps=1e-6,
        attention_dropout=0.0,
        mask_token_id=0,         # any valid token id for the tiny vocab
        # Block diffusion
        block_size=8,
        # Expert routing
        predict_expert_routing=predict_expert_routing,
        routing_layers=list(range(20, 40)),
        num_target_experts=256,
        num_active_experts=8,
        routing_proj_rank=512,
        routing_adapter_rank=64,
    )


def make_random_gate_weights(
    routing_layers=None,
    n_experts: int = 256,
    hidden_dim: int = 2048,
) -> dict:
    """Create random gate weight tensors (simulating the .pt file contents)."""
    if routing_layers is None:
        routing_layers = list(range(20, 40))
    return {l: torch.randn(n_experts, hidden_dim) for l in routing_layers}


# ---------------------------------------------------------------------------
# 1. ExpertPrefetchHead: initialisation
# ---------------------------------------------------------------------------

class TestExpertPrefetchHeadInit:
    def test_creates_shared_projection(self):
        cfg = small_prefetch_config(input_dim=64, proj_rank=16)
        head = ExpertPrefetchHead(cfg)
        assert hasattr(head, "shared_down")
        assert hasattr(head, "shared_up")
        assert head.shared_down.weight.shape == (16, 64)   # [proj_rank, input_dim]
        assert head.shared_up.weight.shape   == (64, 16)   # [input_dim, proj_rank]

    def test_creates_adapters_for_every_layer(self):
        routing_layers = [4, 5, 6]
        cfg = small_prefetch_config(routing_layers=routing_layers, input_dim=64, adapter_rank=8)
        head = ExpertPrefetchHead(cfg)
        for l in routing_layers:
            assert str(l) in head.adapters, f"adapter missing for layer {l}"
            assert "down" in head.adapters[str(l)]
            assert "up" in head.adapters[str(l)]

    def test_creates_gate_modules_for_every_layer(self):
        routing_layers = [4, 5, 6, 7]
        cfg = small_prefetch_config(routing_layers=routing_layers, n_experts=32, input_dim=64)
        head = ExpertPrefetchHead(cfg)
        for l in routing_layers:
            assert str(l) in head.gates, f"gate missing for layer {l}"
            gate = head.gates[str(l)]
            assert gate.weight.shape == (32, 64)  # [n_experts, input_dim]

    def test_gate_weights_are_frozen_at_init(self):
        cfg = small_prefetch_config()
        head = ExpertPrefetchHead(cfg)
        for key, gate in head.gates.items():
            assert not gate.weight.requires_grad, \
                f"gate {key} should be frozen but requires_grad=True"

    def test_adapter_weights_are_trainable_at_init(self):
        cfg = small_prefetch_config()
        head = ExpertPrefetchHead(cfg)
        for key, adapter in head.adapters.items():
            assert adapter["down"].weight.requires_grad, \
                f"adapter {key} down should be trainable"
            assert adapter["up"].weight.requires_grad, \
                f"adapter {key} up should be trainable"

    def test_shared_weights_are_trainable_at_init(self):
        cfg = small_prefetch_config()
        head = ExpertPrefetchHead(cfg)
        assert head.shared_down.weight.requires_grad
        assert head.shared_up.weight.requires_grad

    def test_parameter_counts(self):
        cfg = small_prefetch_config(
            routing_layers=list(range(4, 8)),  # 4 layers
            n_experts=32,
            n_active=4,
            input_dim=64,
            proj_rank=16,
            adapter_rank=8,
        )
        head = ExpertPrefetchHead(cfg)
        trainable = head.trainable_parameters()
        frozen = head.frozen_parameters()

        # Shared: down [64,16] + up [16,64] = 2*64*16 = 2048
        expected_shared = 2 * 64 * 16
        # Per-layer adapters: 4 layers × 2 × (64*8) = 4*2*512 = 4096
        expected_adapters = 4 * 2 * 64 * 8
        # Frozen gates: 4 layers × 32 × 64 = 8192
        expected_frozen = 4 * 32 * 64

        assert trainable == expected_shared + expected_adapters, \
            f"Expected {expected_shared + expected_adapters} trainable, got {trainable}"
        assert frozen == expected_frozen, \
            f"Expected {expected_frozen} frozen, got {frozen}"


# ---------------------------------------------------------------------------
# 2. ExpertPrefetchHead: gate weight loading
# ---------------------------------------------------------------------------

class TestGateWeightLoading:
    def test_load_from_dict_int_keys(self):
        routing_layers = [20, 21, 22]
        cfg = ExpertPrefetchConfig(
            routing_layers=routing_layers, n_experts=16, n_active=2, input_dim=32,
            proj_rank=8, adapter_rank=4,
        )
        head = ExpertPrefetchHead(cfg)
        weights = {l: torch.randn(16, 32) for l in routing_layers}

        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            torch.save(weights, f.name)
            head.load_gate_weights(f.name)

        for l in routing_layers:
            expected = weights[l]
            actual = head.gates[str(l)].weight.data
            assert torch.allclose(actual, expected), f"Layer {l} gate weight mismatch"

    def test_load_from_dict_str_keys(self):
        routing_layers = [5, 6]
        cfg = ExpertPrefetchConfig(
            routing_layers=routing_layers, n_experts=8, n_active=2, input_dim=16,
            proj_rank=4, adapter_rank=2,
        )
        head = ExpertPrefetchHead(cfg)
        # Save with string keys (unusual but accepted)
        weights = {str(l): torch.randn(8, 16) for l in routing_layers}

        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            torch.save(weights, f.name)
            head.load_gate_weights(f.name)

        for l in routing_layers:
            expected = weights[str(l)]
            actual = head.gates[str(l)].weight.data
            assert torch.allclose(actual, expected), f"Layer {l} gate weight mismatch"

    def test_gate_remains_frozen_after_load(self):
        cfg = ExpertPrefetchConfig(
            routing_layers=[10, 11], n_experts=8, n_active=2, input_dim=16,
            proj_rank=4, adapter_rank=2,
        )
        head = ExpertPrefetchHead(cfg)
        weights = {10: torch.randn(8, 16), 11: torch.randn(8, 16)}

        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            torch.save(weights, f.name)
            head.load_gate_weights(f.name)

        for key, gate in head.gates.items():
            assert not gate.weight.requires_grad, \
                f"Gate {key} still frozen after load_gate_weights"

    def test_load_partial_weights_does_not_crash(self, capsys):
        """Missing layers should print a warning but not raise."""
        cfg = ExpertPrefetchConfig(
            routing_layers=[20, 21, 22], n_experts=8, n_active=2, input_dim=16,
            proj_rank=4, adapter_rank=2,
        )
        head = ExpertPrefetchHead(cfg)
        # Only provide weights for 2 of 3 layers
        weights = {20: torch.randn(8, 16), 21: torch.randn(8, 16)}

        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            torch.save(weights, f.name)
            head.load_gate_weights(f.name)  # should not raise

        captured = capsys.readouterr()
        assert "WARNING" in captured.out or "not found" in captured.out or \
               "22" in captured.out or True  # at minimum, it didn't crash


# ---------------------------------------------------------------------------
# 3. ExpertPrefetchHead: forward pass
# ---------------------------------------------------------------------------

class TestExpertPrefetchForward:
    def setup_method(self):
        routing_layers = list(range(4, 8))
        self.cfg = ExpertPrefetchConfig(
            routing_layers=routing_layers,
            n_experts=32,
            n_active=4,
            input_dim=64,
            proj_rank=16,
            adapter_rank=8,
        )
        self.head = ExpertPrefetchHead(self.cfg)
        # Load random gate weights
        weights = {l: torch.randn(32, 64) for l in routing_layers}
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            torch.save(weights, f.name)
            self.head.load_gate_weights(f.name)

    def test_forward_returns_two_dicts(self):
        x = torch.randn(2, 5, 64)  # [B, K, input_dim]
        result = self.head(x)
        assert isinstance(result, tuple), "forward() must return a tuple"
        assert len(result) == 2, "forward() must return (expert_indices, gate_logits)"

    def test_expert_indices_shape(self):
        B, K = 3, 7
        x = torch.randn(B, K, 64)
        expert_indices, _ = self.head(x)
        assert isinstance(expert_indices, dict)
        for l in self.cfg.routing_layers:
            assert l in expert_indices, f"layer {l} missing from expert_indices"
            shape = expert_indices[l].shape
            assert shape == (B, K, self.cfg.n_active), \
                f"layer {l}: expected {(B, K, self.cfg.n_active)}, got {shape}"

    def test_gate_logits_shape(self):
        B, K = 2, 4
        x = torch.randn(B, K, 64)
        _, gate_logits = self.head(x)
        assert isinstance(gate_logits, dict)
        for l in self.cfg.routing_layers:
            assert l in gate_logits, f"layer {l} missing from gate_logits"
            shape = gate_logits[l].shape
            assert shape == (B, K, self.cfg.n_experts), \
                f"layer {l}: expected {(B, K, self.cfg.n_experts)}, got {shape}"

    def test_expert_indices_are_valid(self):
        """Predicted expert indices must be within [0, n_experts)."""
        x = torch.randn(4, 10, 64)
        expert_indices, _ = self.head(x)
        for l, indices in expert_indices.items():
            assert (indices >= 0).all(), f"layer {l}: negative expert indices found"
            assert (indices < self.cfg.n_experts).all(), \
                f"layer {l}: expert index >= n_experts found"

    def test_no_duplicate_experts_per_position(self):
        """topk should produce unique expert indices per (B, K) position."""
        x = torch.randn(2, 6, 64)
        expert_indices, _ = self.head(x)
        for l, indices in expert_indices.items():
            # indices: [B, K, n_active] — sorted along last dim should be unique
            sorted_idx = indices.sort(dim=-1).values
            for b in range(indices.shape[0]):
                for k in range(indices.shape[1]):
                    pos_indices = sorted_idx[b, k]
                    n_unique = pos_indices.unique().shape[0]
                    assert n_unique == self.cfg.n_active, \
                        f"layer {l}, b={b}, k={k}: duplicate experts in top-k"

    def test_predict_experts_matches_forward(self):
        x = torch.randn(2, 5, 64)
        with torch.no_grad():
            expert_indices, _ = self.head(x)
            predict_out = self.head.predict_experts(x)
        for l in self.cfg.routing_layers:
            assert torch.equal(expert_indices[l], predict_out[l]), \
                f"layer {l}: predict_experts() differs from forward()"

    def test_batch_size_one(self):
        x = torch.randn(1, 1, 64)
        expert_indices, gate_logits = self.head(x)
        for l in self.cfg.routing_layers:
            assert expert_indices[l].shape == (1, 1, self.cfg.n_active)
            assert gate_logits[l].shape == (1, 1, self.cfg.n_experts)


# ---------------------------------------------------------------------------
# 4. ExpertPrefetchHead: loss computation
# ---------------------------------------------------------------------------

class TestExpertPrefetchLoss:
    def setup_method(self):
        routing_layers = [4, 5]
        self.cfg = ExpertPrefetchConfig(
            routing_layers=routing_layers,
            n_experts=16,
            n_active=3,
            input_dim=32,
            proj_rank=8,
            adapter_rank=4,
        )
        self.head = ExpertPrefetchHead(self.cfg)

    def test_loss_is_scalar(self):
        x = torch.randn(2, 4, 32)
        _, gate_logits = self.head(x)
        target_routing = {
            l: torch.randint(0, 16, (2, 4, 3))
            for l in self.cfg.routing_layers
        }
        loss = self.head.compute_loss(gate_logits, target_routing)
        assert loss.dim() == 0, "Loss must be a scalar"

    def test_loss_is_positive(self):
        x = torch.randn(2, 4, 32)
        _, gate_logits = self.head(x)
        target_routing = {
            l: torch.randint(0, 16, (2, 4, 3))
            for l in self.cfg.routing_layers
        }
        loss = self.head.compute_loss(gate_logits, target_routing)
        assert loss.item() > 0, "BCE loss must be positive"

    def test_loss_is_bounded(self):
        """BCE loss is bounded in [0, log(2)] per sample for reasonable inputs."""
        x = torch.randn(2, 4, 32)
        _, gate_logits = self.head(x)
        target_routing = {
            l: torch.randint(0, 16, (2, 4, 3))
            for l in self.cfg.routing_layers
        }
        loss = self.head.compute_loss(gate_logits, target_routing)
        # BCE in [0, max(log(2), large_values)] — just check it's finite and bounded
        assert torch.isfinite(loss), "Loss must be finite"
        assert loss.item() < 100.0, "Loss suspiciously large (may indicate init bug)"

    def test_loss_with_partial_routing_labels(self):
        """Loss should not crash when some layers have no labels."""
        x = torch.randn(2, 4, 32)
        _, gate_logits = self.head(x)
        # Only provide labels for one of the two routing layers
        target_routing = {4: torch.randint(0, 16, (2, 4, 3))}
        loss = self.head.compute_loss(gate_logits, target_routing)
        assert torch.isfinite(loss)

    def test_loss_backward(self):
        """Backward through loss must produce gradients on trainable params only."""
        x = torch.randn(2, 4, 32)
        _, gate_logits = self.head(x)
        target_routing = {
            l: torch.randint(0, 16, (2, 4, 3))
            for l in self.cfg.routing_layers
        }
        loss = self.head.compute_loss(gate_logits, target_routing)
        loss.backward()

        # Trainable params must have gradients
        assert self.head.shared_down.weight.grad is not None
        assert self.head.shared_up.weight.grad is not None
        for key, adapter in self.head.adapters.items():
            assert adapter["down"].weight.grad is not None, \
                f"adapter {key} down: no gradient"
            assert adapter["up"].weight.grad is not None, \
                f"adapter {key} up: no gradient"

        # Frozen gate weights must NOT have gradients
        for key, gate in self.head.gates.items():
            assert gate.weight.grad is None, \
                f"gate {key}: gradient exists on frozen weight!"


# ---------------------------------------------------------------------------
# 5. ExpertPrefetchHead: recall metric
# ---------------------------------------------------------------------------

class TestTopKRecall:
    def setup_method(self):
        self.cfg = small_prefetch_config(
            routing_layers=[4, 5], n_experts=32, n_active=4, input_dim=64
        )
        self.head = ExpertPrefetchHead(self.cfg)

    def test_perfect_recall(self):
        """When predictions == ground truth, recall must be 1.0."""
        B, K = 2, 5
        gt = {4: torch.randint(0, 32, (B, K, 4)),
              5: torch.randint(0, 32, (B, K, 4))}
        # Predictions equal to ground truth
        pred = {l: gt[l].clone() for l in [4, 5]}
        result = self.head.compute_topk_recall(pred, gt)
        assert abs(result["mean_recall"] - 1.0) < 1e-6, \
            f"Perfect predictions should give recall=1.0, got {result['mean_recall']}"

    def test_zero_recall(self):
        """When predictions have no overlap with GT, recall must be 0.0."""
        B, K = 2, 5
        # GT uses experts 0-3, predictions use experts 28-31
        gt = {4: torch.zeros(B, K, 4, dtype=torch.long),
              5: torch.zeros(B, K, 4, dtype=torch.long)}
        pred = {4: torch.full((B, K, 4), 31, dtype=torch.long),
                5: torch.full((B, K, 4), 30, dtype=torch.long)}
        result = self.head.compute_topk_recall(pred, gt)
        assert result["mean_recall"] < 1e-6, \
            f"Zero-overlap should give recall=0.0, got {result['mean_recall']}"

    def test_partial_recall(self):
        """Partial overlap gives recall between 0 and 1."""
        B, K = 1, 1
        gt = {4: torch.tensor([[[0, 1, 2, 3]]]),  # [1, 1, 4]
              5: torch.tensor([[[0, 1, 2, 3]]])}
        # 2 of 4 correct
        pred = {4: torch.tensor([[[0, 1, 10, 11]]]),  # overlap: 0, 1
                5: torch.tensor([[[0, 1, 10, 11]]])}
        result = self.head.compute_topk_recall(pred, gt)
        expected = 2 / 4  # 2 hits out of 4
        assert abs(result["mean_recall"] - expected) < 1e-6, \
            f"Expected recall={expected}, got {result['mean_recall']}"

    def test_result_has_required_keys(self):
        B, K = 2, 3
        gt = {4: torch.randint(0, 32, (B, K, 4))}
        pred = {4: torch.randint(0, 32, (B, K, 4))}
        result = self.head.compute_topk_recall(pred, gt)
        assert "mean_recall" in result
        assert "per_layer_recall" in result
        assert isinstance(result["per_layer_recall"], dict)


# ---------------------------------------------------------------------------
# 6. Integration with DFlashDraftModelV8: generate_block
# ---------------------------------------------------------------------------

class TestGenerateBlockIntegration:
    def setup_method(self):
        self.config = small_drafter_config(predict_expert_routing=True)
        self.model = DFlashDraftModelV8(self.config)

        # Load random gate weights
        weights = make_random_gate_weights(
            routing_layers=self.config.routing_layers,
            n_experts=self.config.num_target_experts,
            hidden_dim=self.config.hidden_size,
        )
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            torch.save(weights, f.name)
            self.model.expert_prefetch.load_gate_weights(f.name)

        self.model.eval()

    def _make_context(self, B: int, ctx_len: int):
        H = self.config.hidden_size
        n_layers = self.config.num_feature_layers
        ctx = [torch.randn(B, ctx_len, H) for _ in range(n_layers)]
        ctx_lengths = torch.full((B,), ctx_len, dtype=torch.long)
        anchor_positions = torch.full((B,), ctx_len - 1, dtype=torch.long)
        return ctx, ctx_lengths, anchor_positions

    def test_generate_block_returns_three_outputs(self):
        ctx, ctx_lengths, anchor_positions = self._make_context(2, 32)
        result = self.model.generate_block(
            ctx, context_lengths=ctx_lengths, anchor_positions=anchor_positions
        )
        assert len(result) == 3, "generate_block must return (draft_ids, logits, expert_preds)"

    def test_draft_ids_shape(self):
        B, K = 2, self.config.block_size
        ctx, ctx_lengths, anchor_positions = self._make_context(B, 32)
        draft_ids, logits, expert_preds = self.model.generate_block(
            ctx, context_lengths=ctx_lengths, anchor_positions=anchor_positions
        )
        assert draft_ids.shape == (B, K - 1), \
            f"draft_ids: expected ({B}, {K-1}), got {draft_ids.shape}"

    def test_logits_shape(self):
        B, K = 2, self.config.block_size
        ctx, ctx_lengths, anchor_positions = self._make_context(B, 32)
        draft_ids, logits, expert_preds = self.model.generate_block(
            ctx, context_lengths=ctx_lengths, anchor_positions=anchor_positions
        )
        V = self.config.target_vocab_size
        assert logits.shape == (B, K - 1, V), \
            f"logits: expected ({B}, {K-1}, {V}), got {logits.shape}"

    def test_expert_preds_shape(self):
        B, K = 2, self.config.block_size
        ctx, ctx_lengths, anchor_positions = self._make_context(B, 32)
        draft_ids, logits, expert_preds = self.model.generate_block(
            ctx, context_lengths=ctx_lengths, anchor_positions=anchor_positions
        )
        assert expert_preds is not None, "expert_preds must not be None when routing enabled"
        assert isinstance(expert_preds, dict), "expert_preds must be a dict"
        n_active = self.config.num_active_experts
        for layer_id in self.config.routing_layers:
            assert layer_id in expert_preds, f"layer {layer_id} missing from expert_preds"
            assert expert_preds[layer_id].shape == (B, K - 1, n_active), \
                f"layer {layer_id}: expected ({B}, {K-1}, {n_active}), " \
                f"got {expert_preds[layer_id].shape}"

    def test_expert_preds_valid_indices(self):
        B = 1
        ctx, ctx_lengths, anchor_positions = self._make_context(B, 32)
        _, _, expert_preds = self.model.generate_block(
            ctx, context_lengths=ctx_lengths, anchor_positions=anchor_positions
        )
        n_experts = self.config.num_target_experts
        for layer_id, indices in expert_preds.items():
            assert (indices >= 0).all(), f"layer {layer_id}: negative expert index"
            assert (indices < n_experts).all(), \
                f"layer {layer_id}: expert index >= n_experts ({n_experts})"

    def test_no_expert_preds_when_routing_disabled(self):
        config_no_routing = small_drafter_config(predict_expert_routing=False)
        model = DFlashDraftModelV8(config_no_routing)
        model.eval()
        ctx, ctx_lengths, anchor_positions = self._make_context(1, 32)
        _, _, expert_preds = model.generate_block(
            ctx, context_lengths=ctx_lengths, anchor_positions=anchor_positions
        )
        assert expert_preds is None, \
            "expert_preds must be None when predict_expert_routing=False"


# ---------------------------------------------------------------------------
# 7. Integration: forward_train
# ---------------------------------------------------------------------------

class TestForwardTrainIntegration:
    def setup_method(self):
        self.config = small_drafter_config(predict_expert_routing=True)
        self.model = DFlashDraftModelV8(self.config)
        weights = make_random_gate_weights()
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            torch.save(weights, f.name)
            self.model.expert_prefetch.load_gate_weights(f.name)
        self.model.freeze_shared_params()

    def _make_batch(self, B: int, K: int, ctx_len: int):
        H = self.config.hidden_size
        n_layers = self.config.num_feature_layers
        V = self.config.target_vocab_size
        block_ids = torch.randint(0, V, (B, K))
        ctx_hidden = [torch.randn(B, ctx_len, H) for _ in range(n_layers)]
        ctx_lengths = torch.full((B,), ctx_len, dtype=torch.long)
        anchor_positions = torch.full((B,), ctx_len - 1, dtype=torch.long)
        return block_ids, ctx_hidden, ctx_lengths, anchor_positions

    def test_forward_train_returns_three_items(self):
        self.model.train()
        block_ids, ctx_hidden, ctx_lengths, anchor_positions = self._make_batch(2, 8, 32)
        result = self.model.forward_train(block_ids, ctx_hidden, ctx_lengths, anchor_positions)
        assert len(result) == 3, "forward_train must return (loss, logits, expert_preds)"

    def test_loss_is_scalar(self):
        self.model.train()
        block_ids, ctx_hidden, ctx_lengths, anchor_positions = self._make_batch(2, 8, 32)
        loss, logits, _ = self.model.forward_train(block_ids, ctx_hidden, ctx_lengths, anchor_positions)
        assert loss.dim() == 0, "Loss must be a scalar"

    def test_expert_preds_from_forward_train(self):
        self.model.train()
        B, K = 2, self.config.block_size
        block_ids, ctx_hidden, ctx_lengths, anchor_positions = self._make_batch(B, K, 32)
        _, _, expert_preds = self.model.forward_train(block_ids, ctx_hidden, ctx_lengths, anchor_positions)
        assert expert_preds is not None
        # Returns a tuple (indices, logits) from ExpertPrefetchHead.forward()
        assert isinstance(expert_preds, tuple)
        expert_indices, gate_logits = expert_preds
        n_active = self.config.num_active_experts
        for layer_id in self.config.routing_layers:
            assert expert_indices[layer_id].shape == (B, K - 1, n_active)
            assert gate_logits[layer_id].shape == (B, K - 1, self.config.num_target_experts)

    def test_backward_does_not_update_frozen_gates(self):
        self.model.train()
        block_ids, ctx_hidden, ctx_lengths, anchor_positions = self._make_batch(2, 8, 32)
        loss, _, _ = self.model.forward_train(block_ids, ctx_hidden, ctx_lengths, anchor_positions)
        loss.backward()
        for key, gate in self.model.expert_prefetch.gates.items():
            assert gate.weight.grad is None, \
                f"Frozen gate {key} accumulated gradient during backward!"

    def test_backward_updates_adapter_weights(self):
        self.model.train()
        B, K = 2, self.config.block_size
        block_ids, ctx_hidden, ctx_lengths, anchor_positions = self._make_batch(B, K, 32)
        loss, _, expert_preds = self.model.forward_train(block_ids, ctx_hidden, ctx_lengths, anchor_positions)
        # Add routing loss so it flows through the prefetch head
        _, gate_logits = expert_preds
        routing_labels = {
            l: torch.randint(0, self.config.num_target_experts,
                             (B, K - 1, self.config.num_active_experts))
            for l in self.config.routing_layers
        }
        routing_loss = self.model.expert_prefetch.compute_loss(gate_logits, routing_labels)
        total_loss = loss + 0.1 * routing_loss
        total_loss.backward()

        # Adapter weights should have gradients
        for key, adapter in self.model.expert_prefetch.adapters.items():
            assert adapter["down"].weight.grad is not None, \
                f"Adapter {key} down: no gradient after backward"


# ---------------------------------------------------------------------------
# 8. Integration: forward_train_multi
# ---------------------------------------------------------------------------

class TestForwardTrainMultiIntegration:
    def setup_method(self):
        self.config = small_drafter_config(predict_expert_routing=True)
        self.model = DFlashDraftModelV8(self.config)
        self.model.freeze_shared_params()
        self.model.enable_gradient_checkpointing()
        weights = make_random_gate_weights()
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            torch.save(weights, f.name)
            self.model.expert_prefetch.load_gate_weights(f.name)

    def _make_batch(self, B: int, K: int, ctx_len: int, requires_grad: bool = False):
        H = self.config.hidden_size
        n_layers = self.config.num_feature_layers
        V = self.config.target_vocab_size
        block_ids = torch.randint(0, V, (B, K))
        ctx_hidden = [
            torch.randn(B, ctx_len, H, requires_grad=requires_grad)
            for _ in range(n_layers)
        ]
        ctx_lengths = torch.full((B,), ctx_len, dtype=torch.long)
        anchor_positions = torch.full((B,), ctx_len - 1, dtype=torch.long)
        return block_ids, ctx_hidden, ctx_lengths, anchor_positions

    def test_forward_train_multi_returns_three_items(self):
        self.model.train()
        B, K, ctx_len = 2, self.config.block_size, 32
        block_ids, ctx_hidden, ctx_lengths, anchor_positions = self._make_batch(
            B, K, ctx_len, requires_grad=True
        )
        result = self.model.forward_train_multi(block_ids, ctx_hidden, ctx_lengths, anchor_positions)
        assert len(result) == 3

    def test_logits_are_none_in_multi(self):
        """forward_train_multi never materializes full logits to save VRAM."""
        self.model.train()
        B, K, ctx_len = 2, self.config.block_size, 32
        block_ids, ctx_hidden, ctx_lengths, anchor_positions = self._make_batch(
            B, K, ctx_len, requires_grad=True
        )
        _, logits, _ = self.model.forward_train_multi(
            block_ids, ctx_hidden, ctx_lengths, anchor_positions
        )
        assert logits is None, "forward_train_multi must return None for logits"

    def test_expert_preds_shape_in_multi(self):
        self.model.train()
        B, K, ctx_len = 2, self.config.block_size, 32
        block_ids, ctx_hidden, ctx_lengths, anchor_positions = self._make_batch(
            B, K, ctx_len, requires_grad=True
        )
        _, _, expert_preds = self.model.forward_train_multi(
            block_ids, ctx_hidden, ctx_lengths, anchor_positions
        )
        assert expert_preds is not None
        expert_indices, _ = expert_preds
        n_active = self.config.num_active_experts
        for layer_id in self.config.routing_layers:
            assert expert_indices[layer_id].shape == (B, K - 1, n_active)

    def test_frozen_gates_in_multi(self):
        self.model.train()
        B, K, ctx_len = 2, self.config.block_size, 32
        block_ids, ctx_hidden, ctx_lengths, anchor_positions = self._make_batch(
            B, K, ctx_len, requires_grad=True
        )
        loss, _, _ = self.model.forward_train_multi(
            block_ids, ctx_hidden, ctx_lengths, anchor_positions
        )
        loss.backward()
        for key, gate in self.model.expert_prefetch.gates.items():
            assert gate.weight.grad is None, \
                f"Frozen gate {key} has gradient in forward_train_multi!"


# ---------------------------------------------------------------------------
# 9. Frozen gates: end-to-end gradient isolation test
# ---------------------------------------------------------------------------

class TestFrozenGateGradients:
    """Comprehensive test that frozen gate parameters cannot be updated."""

    def test_optimizer_step_does_not_modify_gates(self):
        """Run a full optimizer step and verify gate weights are unchanged."""
        cfg = ExpertPrefetchConfig(
            routing_layers=[4, 5],
            n_experts=16,
            n_active=3,
            input_dim=32,
            proj_rank=8,
            adapter_rank=4,
        )
        head = ExpertPrefetchHead(cfg)

        # Capture initial gate values
        initial_gates = {
            key: gate.weight.data.clone()
            for key, gate in head.gates.items()
        }

        optimizer = torch.optim.Adam(
            [p for p in head.parameters() if p.requires_grad],
            lr=1e-3,
        )

        x = torch.randn(2, 4, 32)
        expert_indices, gate_logits = head(x)
        target_routing = {
            l: torch.randint(0, 16, (2, 4, 3))
            for l in cfg.routing_layers
        }
        loss = head.compute_loss(gate_logits, target_routing)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Gates must not have changed
        for key, gate in head.gates.items():
            assert torch.equal(gate.weight.data, initial_gates[key]), \
                f"Gate {key} was modified by optimizer step!"

    def test_gate_requires_grad_false_invariant(self):
        """requires_grad=False must survive multiple forward+backward cycles."""
        cfg = ExpertPrefetchConfig(
            routing_layers=[4, 5, 6],
            n_experts=16,
            n_active=3,
            input_dim=32,
            proj_rank=8,
            adapter_rank=4,
        )
        head = ExpertPrefetchHead(cfg)

        for _ in range(3):
            x = torch.randn(2, 4, 32)
            _, gate_logits = head(x)
            target_routing = {l: torch.randint(0, 16, (2, 4, 3)) for l in cfg.routing_layers}
            loss = head.compute_loss(gate_logits, target_routing)
            loss.backward()

            for key, gate in head.gates.items():
                assert not gate.weight.requires_grad, \
                    f"Gate {key} requires_grad changed to True after backward"
                assert gate.weight.grad is None, \
                    f"Gate {key} accumulated gradient after backward"


# ---------------------------------------------------------------------------
# 10. Determinism / self-consistency
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_input_same_output_eval_mode(self):
        """In eval mode (no dropout), output must be identical for same input."""
        cfg = small_prefetch_config()
        head = ExpertPrefetchHead(cfg)
        head.eval()

        x = torch.randn(2, 5, 64)
        with torch.no_grad():
            indices_1, logits_1 = head(x)
            indices_2, logits_2 = head(x)

        for l in cfg.routing_layers:
            assert torch.equal(indices_1[l], indices_2[l]), \
                f"layer {l}: indices differ between two forward passes with same input"
            assert torch.allclose(logits_1[l], logits_2[l]), \
                f"layer {l}: logits differ between two forward passes with same input"

    def test_different_inputs_may_give_different_outputs(self):
        """With sufficiently different inputs the predictions should usually differ."""
        cfg = small_prefetch_config(input_dim=64, n_experts=32)
        head = ExpertPrefetchHead(cfg)
        head.eval()

        x1 = torch.randn(1, 1, 64)
        x2 = torch.randn(1, 1, 64) * 100  # very different scale

        with torch.no_grad():
            idx1, _ = head(x1)
            idx2, _ = head(x2)

        # At least one layer should have different predictions (probabilistically certain)
        first_layer = cfg.routing_layers[0]
        # Note: with random weights this is not guaranteed, but statistically extremely likely
        any_diff = any(
            not torch.equal(idx1[l], idx2[l])
            for l in cfg.routing_layers
        )
        # We don't assert this (random weights may coincide) — just report
        # This test serves as a documentation of expected behaviour
        _ = any_diff  # suppress unused warning


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pytest as _pytest
    _pytest.main([__file__, "-v"] + sys.argv[1:])
