#!/usr/bin/env python3
"""Merge all prompt files, deduplicate, shuffle → data/all_prompts.jsonl"""
import json
import hashlib
import random
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"

prompt_files = [
    DATA / "bootstrap_prompts.jsonl",
    DATA / "diverse_prompts_v2.jsonl",
    DATA / "prompts.jsonl",
    DATA / "prompts_v1_500.jsonl",
]

seen = set()
prompts = []

for pf in prompt_files:
    if not pf.exists():
        continue
    with open(pf) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("text", "")
            h = hashlib.md5(text.encode()).hexdigest()
            if h not in seen and len(text) > 10:
                seen.add(h)
                prompts.append(obj)

random.seed(42)
random.shuffle(prompts)

out = DATA / "all_prompts.jsonl"
with open(out, "w") as f:
    for p in prompts:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")

print(f"Merged {len(prompts)} unique prompts → {out}")
