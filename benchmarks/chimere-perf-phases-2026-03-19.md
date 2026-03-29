# Chimere-DeltaNet Performance — Final Status (2026-03-19)

Branch: `rewrite-cudarc` | Tag: `final-session-2026-03-19`

## Performance Journey

```
Mar 14   21.0 tok/s   initial DeltaNet (Candle)
Mar 15   59.0 tok/s   bench-only (ggml GPU FFI)
Mar 18   37.8 tok/s   cudarc rewrite, server validated
Mar 19   57.0 tok/s   full optimizations (cudarc path)
Mar 19   93.0 tok/s   libllama FFI backend (parité ik_llama)
```

## ALL Phases COMPLETED

| Phase | Commit | Tok/s | Status |
|-------|--------|-------|--------|
| Phase 0 (quick wins) | ef56e07 | 38.5 | ✅ |
| Phase 1.1 (MoE micro) | c7b255e | 39.6 | ✅ |
| Phase 1.2 (GPU-resident MoE) | 06ec181 | 51.8 | ✅ |
| Phase 2.2 (fused GDN) | f35c1c8 | 39.3 | ✅ |
| Phase 2.4 (fused RMSNorm) | 6663349 | 39.2 | ✅ |
| Phase 0.5 (cubin) | 68b0a52 | 52.9 | ✅ |
| Phase 3.1 (CUDA Graphs 36) | 4604e01 | 57.0 | ✅ |
| In-place DeltaNet | c72b59d | — | ✅ |
| Cached raw pointers | 741584e | — | ✅ |
| Fused Q8_1+GEMV | c38d0b1 | — | ✅ |
| Batch prefill V2 | 9363cfc | — | ✅ |
| Flash Attn prefill | c8cae1c | — | ✅ (infra, divergence à debugger) |
| Residual zero fix | da3ce91 | — | ✅ (root cause garbage gen) |
| **libllama FFI** | **42ae8a5** | **93** | **✅ PARITÉ** |

## 3 Backends disponibles

| Backend | Toggle | Tok/s | Features |
|---------|--------|-------|----------|
| libllama | `CHIMERE_LLAMA_BACKEND=1` | 93 | Parité ik_llama, MTP possible |
| cudarc | `CHIMERE_CUDARC_FORWARD=1` | 57 | Custom DeltaNet, Engram possible |
| candle | default | ~18 | Complet mais lent |

## Next: MTP (×1.7-1.9 throughput)
- ik_llama a MTP natif, blocage = GDN recurrent state save/restore
- DFlash drafter disponible sur cette machine
- Engram après MTP (killer feature unique)
