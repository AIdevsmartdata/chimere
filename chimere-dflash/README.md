# Chimère-DFlash

Block diffusion drafter for speculative decoding, targeting Qwen3.5-35B-A3B on RTX 5060 Ti 16GB.

## Architecture

```
Qwen3.5 (target, frozen) -> hidden states -> Feature Fusion -> KV Injection
                                                    |
                              Noise -> [5-layer Bidir Transformer] -> Logits
                                            (denoiser)
                                                    |
                              16 draft tokens -> Verify with Qwen3.5 -> Accept streak
```

## Quick Start

```bash
python -m pytest tests/ -v   # Run tests
```

## References

- DFlash: arXiv 2602.06036
- BD3LM: Block Discrete Denoising Diffusion
- Qwen3.5: huggingface.co/Qwen/Qwen3.5-35B-A3B
