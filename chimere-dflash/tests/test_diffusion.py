import torch
from chimere.config import DFlashConfig
from chimere.diffusion import MaskDiffusion, GaussianDiffusion


def test_mask_apply_shapes():
    """Masking should preserve shapes and create a valid mask."""
    config = DFlashConfig()
    diffusion = MaskDiffusion(config)
    B, S, H = 2, 16, config.drafter_hidden_size
    x_0 = torch.randn(B, S, H)
    mask_ratios = torch.tensor([0.5, 0.75])
    x_masked, mask = diffusion.apply_mask(x_0, mask_ratios)
    assert x_masked.shape == (B, S, H)
    assert mask.shape == (B, S)
    assert mask.dtype == torch.bool
    assert mask.any()


def test_mask_anchor_token():
    """With anchor_first_token=True, position 0 should never be masked."""
    config = DFlashConfig(anchor_first_token=True)
    diffusion = MaskDiffusion(config)
    B, S, H = 4, 16, config.drafter_hidden_size
    x_0 = torch.randn(B, S, H)
    # Even with 100% mask ratio, position 0 stays unmasked
    mask_ratios = torch.ones(B)
    x_masked, mask = diffusion.apply_mask(x_0, mask_ratios)
    assert not mask[:, 0].any(), "Position 0 (anchor) should never be masked"
    assert mask[:, 1:].all(), "All other positions should be masked at ratio=1.0"


def test_mask_no_anchor():
    """With anchor_first_token=False, position 0 can be masked."""
    config = DFlashConfig(anchor_first_token=False)
    diffusion = MaskDiffusion(config)
    B, S, H = 4, 16, config.drafter_hidden_size
    x_0 = torch.randn(B, S, H)
    mask_ratios = torch.ones(B)
    x_masked, mask = diffusion.apply_mask(x_0, mask_ratios)
    assert mask.all(), "All positions including 0 should be masked"


def test_mask_full_mask_with_anchor():
    """Full mask with anchor should leave position 0 unmasked."""
    config = DFlashConfig(anchor_first_token=True)
    diffusion = MaskDiffusion(config)
    B, S, H = 2, 16, config.drafter_hidden_size
    anchor_embeds = torch.randn(B, H)
    x_masked, mask = diffusion.apply_full_mask(
        B, S, H, "cpu", torch.float32, anchor_embeds=anchor_embeds
    )
    assert x_masked.shape == (B, S, H)
    assert not mask[:, 0].any(), "Anchor position should be unmasked"
    assert mask[:, 1:].all(), "Non-anchor positions should be masked"
    # Verify anchor embedding is used
    assert torch.allclose(x_masked[:, 0, :], anchor_embeds)


def test_mask_full_mask_no_anchor():
    """Full mask without anchor embeds should mask all positions."""
    config = DFlashConfig()
    diffusion = MaskDiffusion(config)
    B, S, H = 2, 16, config.drafter_hidden_size
    x_masked, mask = diffusion.apply_full_mask(B, S, H, "cpu", torch.float32)
    assert mask.all(), "All positions should be masked when no anchor provided"


def test_mask_ratio_sampling():
    """Mask ratios should be in valid range."""
    config = DFlashConfig()
    diffusion = MaskDiffusion(config)
    ratios = diffusion.sample_mask_ratio(100, "cpu")
    assert ratios.shape == (100,)
    assert (ratios >= 0.1).all()
    assert (ratios <= 1.0).all()


# Legacy tests for backward compat
def test_gaussian_q_sample_shapes():
    config = DFlashConfig()
    diffusion = GaussianDiffusion(config)
    B, S, H = 2, 16, 2048
    x_0 = torch.randn(B, S, H)
    t = torch.randint(0, 1000, (B,))
    x_t, noise = diffusion.q_sample(x_0, t)
    assert x_t.shape == (B, S, H)
    assert noise.shape == (B, S, H)


def test_gaussian_noise_preserves_mean():
    """At t=0, noised signal should be very close to original."""
    config = DFlashConfig()
    diffusion = GaussianDiffusion(config)
    x_0 = torch.randn(2, 16, 256)
    t = torch.zeros(2, dtype=torch.long)
    x_t, _ = diffusion.q_sample(x_0, t)
    assert torch.allclose(x_0, x_t, atol=0.05)
