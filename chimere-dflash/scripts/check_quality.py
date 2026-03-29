#!/usr/bin/env python3
"""Quick quality check on extracted features."""
import numpy as np
import json
import random
from pathlib import Path

d = Path("data/features_q5")
dirs = sorted([x for x in d.iterdir() if x.is_dir()])
n = len(dirs)
print(f"Total samples: {n}")

random.seed(42)
checks = random.sample(range(n), min(100, n))

ok = 0
bad_size = 0
bad_nan = 0
bad_block = 0
anchor_positions = []
seq_lens = []
sources = set()

for idx in checks:
    sd = dirs[idx]
    ah = sd / "anchor_hidden.bin"
    bt = sd / "block_tokens.bin"
    mf = sd / "metadata.json"

    if not ah.exists() or not bt.exists() or not mf.exists():
        bad_size += 1
        continue

    if ah.stat().st_size != 20480:
        bad_size += 1
        continue

    if bt.stat().st_size != 64:
        bad_block += 1
        continue

    h = np.fromfile(ah, dtype=np.float16).reshape(5, 2048)
    if np.any(np.isnan(h)) or np.any(np.isinf(h)):
        bad_nan += 1
        continue

    meta = json.load(open(mf))
    anchor_positions.append(meta["anchor_pos"])
    seq_lens.append(meta["seq_len"])
    src = meta.get("source_id", "")
    prefix = src.split("_")[0] if "_" in src else src
    sources.add(prefix)
    ok += 1

print(f"Checked: {len(checks)} | OK: {ok} | bad_size: {bad_size} | bad_nan: {bad_nan} | bad_block: {bad_block}")
print(f"Anchor pos: min={min(anchor_positions)}, max={max(anchor_positions)}, mean={np.mean(anchor_positions):.0f}")
print(f"Seq len:    min={min(seq_lens)}, max={max(seq_lens)}, mean={np.mean(seq_lens):.0f}")
print(f"Sources:    {sorted(sources)}")

# Hidden state stats
stds = []
means = []
for idx in checks[:10]:
    sd = dirs[idx]
    h = np.fromfile(sd / "anchor_hidden.bin", dtype=np.float16).reshape(5, 2048).astype(np.float32)
    stds.append(h.std())
    means.append(h.mean())
print(f"Hidden stats (10 samples): mean={np.mean(means):.4f}, std={np.mean(stds):.4f}")
