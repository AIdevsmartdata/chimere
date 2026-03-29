import torch
from chimere.config import DFlashConfig
from chimere.modeling import DFlashDraftModel


def _small_config():
    return DFlashConfig(
        target_vocab_size=1000,
        drafter_hidden_size=256,
        target_hidden_size=256,
        drafter_num_heads=4,
        drafter_head_dim=64,
        drafter_intermediate_size=512,
        drafter_num_layers=2,
        num_feature_layers=2,
        fusion_dim=256,
    )


def test_forward_train():
    config = _small_config()
    model = DFlashDraftModel(config)
    B, S = 2, 16
    input_ids = torch.randint(0, 1000, (B, S))
    hidden_states = [torch.randn(B, 32, 256) for _ in range(2)]
    loss, logits = model.forward_train(input_ids, hidden_states)
    assert logits.shape == (B, S, 1000)
    assert loss.dim() == 0


def test_backward():
    config = _small_config()
    model = DFlashDraftModel(config)
    B, S = 2, 16
    input_ids = torch.randint(0, 1000, (B, S))
    hidden_states = [torch.randn(B, 32, 256) for _ in range(2)]
    loss, _ = model.forward_train(input_ids, hidden_states)
    loss.backward()
    # Check gradients exist on trainable params
    grads = [p.grad for p in model.layers.parameters() if p.grad is not None]
    assert len(grads) > 0


def test_generate_block():
    config = _small_config()
    model = DFlashDraftModel(config).eval()
    B = 2
    hidden_states = [torch.randn(B, 32, 256) for _ in range(2)]
    draft_ids, logits = model.generate_block(hidden_states)
    assert draft_ids.shape == (B, 16)
    assert logits.shape == (B, 16, 1000)


def test_freeze_shared_params():
    config = _small_config()
    model = DFlashDraftModel(config)
    model.freeze_shared_params()
    assert not model.embed_tokens.weight.requires_grad
    assert not model.lm_head.weight.requires_grad
    # Layers should still be trainable
    assert any(p.requires_grad for p in model.layers.parameters())
