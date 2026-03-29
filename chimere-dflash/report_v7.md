# DFlash v7 — Training Report (RoPE + Absolute Positions)

## Training Log (final epochs)
```
Epoch 1/6 | train_loss=5.8148 | val_loss=5.2241 | val_acc=0.1462 | lr=5.75e-04 | 1087.1s | 1389 tok/s
  -> New best model saved (val_loss=5.2241)
Epoch 2/6 | train_loss=4.7443 | val_loss=4.7600 | val_acc=0.1715 | lr=4.72e-04 | 1088.2s | 1388 tok/s
  -> New best model saved (val_loss=4.7600)
Epoch 3/6 | train_loss=3.9152 | val_loss=4.3920 | val_acc=0.1970 | lr=3.19e-04 | 1087.7s | 1388 tok/s
  -> New best model saved (val_loss=4.3920)
Epoch 4/6 | train_loss=3.0855 | val_loss=4.1445 | val_acc=0.2311 | lr=1.61e-04 | 1087.8s | 1388 tok/s
  -> New best model saved (val_loss=4.1445)
Epoch 5/6 | train_loss=2.4041 | val_loss=4.0184 | val_acc=0.2575 | lr=4.34e-05 | 1087.4s | 1389 tok/s
  -> New best model saved (val_loss=4.0184)
Epoch 6/6 | train_loss=2.0168 | val_loss=4.0380 | val_acc=0.2620 | lr=1.85e-10 | 1089.3s | 1386 tok/s
```

## τ Benchmark (best checkpoint)
```
Loading checkpoint: checkpoints_v7_98k/best.pt
  block_size=16, layers=8, features=5, rope_theta=10000000
  Evaluating 1000/99347 samples

  [100/1000] τ=0.27% | avg_accepted=0.0/15 | draft=5.6ms
  [200/1000] τ=0.30% | avg_accepted=0.1/15 | draft=2.9ms
  [300/1000] τ=0.22% | avg_accepted=0.0/15 | draft=2.9ms
  [400/1000] τ=0.20% | avg_accepted=0.0/15 | draft=2.9ms
  [500/1000] τ=0.24% | avg_accepted=0.1/15 | draft=2.9ms
  [600/1000] τ=0.21% | avg_accepted=0.0/15 | draft=2.9ms
  [700/1000] τ=0.24% | avg_accepted=0.1/15 | draft=2.9ms
  [800/1000] τ=0.25% | avg_accepted=0.1/15 | draft=2.9ms
  [900/1000] τ=0.24% | avg_accepted=0.0/15 | draft=2.9ms
  [1000/1000] τ=0.25% | avg_accepted=0.1/15 | draft=2.9ms

============================================================
 DFlash v7 — Offline τ Benchmark (RoPE + Absolute Pos)
============================================================
  Checkpoint:       checkpoints_v7_98k/best.pt
  Samples eval:     1000
  Block size:       16
  Temperature:      0.0
  RoPE theta:       10000000

  Total drafted:    15000
  Total accepted:   38
  τ (accept rate):  0.25%
  Avg accepted:     0.0 / 15
  Avg draft time:   3.2ms
  Total time:       13.9s

  Tokens/target_call:  1.0
  Theoretical max:     16x
  Est. real speedup:   0.92x
    (assuming target=25ms/token, draft=3ms/block)
============================================================

  Acceptance distribution (tokens accepted per block):
     0:  967 ( 96.7%) ################################################
     1:   29 (  2.9%) #
     2:    3 (  0.3%) 
     3:    1 (  0.1%) 

```

## Comparaison v6 vs v7
| Metric | v6 (98K) | v7 (98K) |
|--------|----------|----------|
| Best val_loss | 5.06 | 4.0184 |
| val_acc | 8.3% | 0.2620 |
| τ (accept rate) | 2.7% |  0.25% |
| Avg accepted/block | 0.4 |     0.0 / 15 |

## Verdict

*(auto-generated — check benchmark_v7.log for full details)*

---
Generated: 2026-03-04 03:19
