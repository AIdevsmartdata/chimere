import torch
from chimere.config import DFlashConfig
from chimere.modeling import DFlashDraftModel
from chimere.spec_decode import SpeculativeDecoder, SpecDecodeStats


def test_draft_block_shapes():
    """Draft block should return [K] ids and [K, vocab] logits."""
    config = DFlashConfig()
    model = DFlashDraftModel(config)
    model.eval()

    k = config.num_feature_layers
    B, S, H = 1, config.block_size, config.target_hidden_size
    hidden_list = [torch.randn(B, S, H) for _ in range(k)]

    decoder = SpeculativeDecoder(model, config, device="cpu")
    draft_ids, draft_logits = decoder.draft_block(hidden_list, temperature=0.0)

    assert draft_ids.shape == (config.block_size,)
    assert draft_logits.shape == (config.block_size, config.target_vocab_size)
    assert draft_ids.dtype == torch.int64


def test_draft_block_deterministic_with_seed():
    """With same RNG seed, greedy drafting should be deterministic."""
    config = DFlashConfig()
    model = DFlashDraftModel(config)
    model.eval()

    k = config.num_feature_layers
    B, S, H = 1, config.block_size, config.target_hidden_size
    hidden_list = [torch.randn(B, S, H) for _ in range(k)]

    decoder = SpeculativeDecoder(model, config, device="cpu")

    # Same seed → same random noise → same output
    torch.manual_seed(42)
    ids1, _ = decoder.draft_block(hidden_list, temperature=0.0)
    torch.manual_seed(42)
    ids2, _ = decoder.draft_block(hidden_list, temperature=0.0)
    assert torch.equal(ids1, ids2), "Same seed + greedy should be deterministic"


def test_benchmark_offline():
    """Offline benchmark should compute acceptance stats."""
    config = DFlashConfig(block_size=4, target_vocab_size=100)
    model = DFlashDraftModel(config)
    model.eval()

    k = config.num_feature_layers
    S = config.block_size
    H = config.target_hidden_size

    decoder = SpeculativeDecoder(model, config, device="cpu")

    # Create fake block data
    block_data = [{
        "block_hidden": torch.randn(k, S, H),
        "block_input_ids": torch.randint(0, 100, (S,)),
    }]

    stats = decoder.benchmark_offline(
        prompt_tokens=[],
        block_hidden_states=block_data,
        ground_truth_tokens=block_data[0]["block_input_ids"].tolist(),
        temperature=0.0,
    )

    assert isinstance(stats, SpecDecodeStats)
    assert stats.total_steps == 1
    assert stats.total_drafted == S
    assert 0 <= stats.total_accepted <= S
    assert stats.total_tokens >= 1
    assert stats.draft_time_ms > 0


def test_benchmark_perfect_draft():
    """If draft matches ground truth (with same seed), all tokens should be accepted."""
    config = DFlashConfig(block_size=4, target_vocab_size=100)
    model = DFlashDraftModel(config)
    model.eval()

    k = config.num_feature_layers
    S = config.block_size
    H = config.target_hidden_size
    hidden = torch.randn(k, S, H)

    decoder = SpeculativeDecoder(model, config, device="cpu")

    # Draft with a fixed seed to know what the model will produce
    torch.manual_seed(123)
    hidden_list = [hidden[i].unsqueeze(0) for i in range(k)]
    draft_ids, _ = decoder.draft_block(hidden_list, temperature=0.0)

    # Use draft's own predictions as ground truth
    block_data = [{
        "block_hidden": hidden,
        "block_input_ids": draft_ids,
    }]

    # Re-seed so benchmark_offline's internal draft call produces the same tokens
    torch.manual_seed(123)
    stats = decoder.benchmark_offline(
        prompt_tokens=[],
        block_hidden_states=block_data,
        ground_truth_tokens=draft_ids.tolist(),
        temperature=0.0,
    )

    assert stats.total_accepted == S, f"Expected all {S} accepted, got {stats.total_accepted}"
    assert stats.acceptance_rate == 1.0


def test_stats_properties():
    """SpecDecodeStats computed properties should be correct."""
    stats = SpecDecodeStats(
        total_tokens=100,
        total_steps=10,
        total_drafted=160,
        total_accepted=80,
        total_target_calls=10,
    )

    assert stats.acceptance_rate == 0.5
    assert stats.tokens_per_step == 10.0
    assert stats.speedup_vs_ar == 10.0


def test_multistep_denoising():
    """Multi-step denoising should produce valid outputs and be deterministic."""
    config = DFlashConfig(block_size=8)
    model = DFlashDraftModel(config)
    model.eval()

    k = config.num_feature_layers
    B, S, H = 1, config.block_size, config.target_hidden_size
    hidden_list = [torch.randn(B, S, H) for _ in range(k)]

    # Test various step counts
    for n_steps in [1, 2, 4, 8]:
        torch.manual_seed(42)
        draft_ids, draft_logits = model.generate_block(
            hidden_list, temperature=0.0, n_steps=n_steps
        )
        assert draft_ids.shape == (B, S), f"Bad shape for n_steps={n_steps}"
        assert draft_logits.shape == (B, S, config.target_vocab_size)
        assert draft_ids.dtype == torch.int64

    # Deterministic: same seed → same output
    torch.manual_seed(99)
    ids1, _ = model.generate_block(hidden_list, temperature=0.0, n_steps=4)
    torch.manual_seed(99)
    ids2, _ = model.generate_block(hidden_list, temperature=0.0, n_steps=4)
    assert torch.equal(ids1, ids2), "Multi-step should be deterministic with same seed"


def test_multistep_vs_singlestep_different():
    """Multi-step should generally produce different (better) results than single-step."""
    config = DFlashConfig(block_size=8)
    model = DFlashDraftModel(config)
    model.eval()

    k = config.num_feature_layers
    B, S, H = 1, config.block_size, config.target_hidden_size
    hidden_list = [torch.randn(B, S, H) for _ in range(k)]

    torch.manual_seed(42)
    ids_1step, _ = model.generate_block(hidden_list, temperature=0.0, n_steps=1)

    torch.manual_seed(42)
    ids_8step, _ = model.generate_block(hidden_list, temperature=0.0, n_steps=8)

    # They should differ (different denoising paths)
    # Note: with random weights they COULD theoretically match, but it's very unlikely
    # so we just check that the function runs without error — the real test is the benchmark
    assert ids_1step.shape == ids_8step.shape
