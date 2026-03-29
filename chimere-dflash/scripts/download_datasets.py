#!/usr/bin/env python3
"""Download and prepare diverse prompt dataset for DFlash v6 training.

Sources (z-lab recipe + diversity):
  - Nemotron Post-Training v2: code, math, chat, stem
  - evol-codealpaca-v1: evolved code instructions
  - CodeFeedback-Filtered-Instruction: diverse code

Target: 100K prompts in JSONL format for generate_prompts_v6.py
"""

import json
import os
import random
import sys
from pathlib import Path

# Check dependencies
try:
    from datasets import load_dataset
except ImportError:
    print("pip install datasets")
    sys.exit(1)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "prompts_v6"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)


def extract_prompt(example, dataset_name):
    """Extract a prompt string from various dataset formats."""
    # Nemotron v2 SFT format: has 'input' and 'output' or conversations
    if "input" in example and example["input"]:
        return example["input"].strip()
    if "instruction" in example and example["instruction"]:
        prompt = example["instruction"].strip()
        if "input" in example and example["input"] and example["input"].strip():
            prompt += "\n\n" + example["input"].strip()
        return prompt
    if "query" in example and example["query"]:
        return example["query"].strip()
    if "prompt" in example and example["prompt"]:
        if isinstance(example["prompt"], list):
            # Conversations format
            for msg in example["prompt"]:
                if msg.get("role") == "user":
                    return msg["content"].strip()
        return str(example["prompt"]).strip()
    if "conversations" in example and example["conversations"]:
        convs = example["conversations"]
        for msg in convs:
            role = msg.get("from", msg.get("role", ""))
            content = msg.get("value", msg.get("content", ""))
            if role in ("human", "user") and content:
                return content.strip()
    if "text" in example and example["text"]:
        return example["text"].strip()
    if "messages" in example and example["messages"]:
        for msg in example["messages"]:
            if msg.get("role") == "user":
                return msg["content"].strip()
    return None


def extract_response(example):
    """Extract response if available (for pre-generated responses)."""
    if "output" in example and example["output"]:
        return str(example["output"]).strip()
    if "answer" in example and example["answer"]:
        return str(example["answer"]).strip()
    if "response" in example and example["response"]:
        return str(example["response"]).strip()
    if "conversations" in example and example["conversations"]:
        convs = example["conversations"]
        for msg in convs:
            role = msg.get("from", msg.get("role", ""))
            content = msg.get("value", msg.get("content", ""))
            if role in ("gpt", "assistant") and content:
                return content.strip()
    if "messages" in example and example["messages"]:
        for msg in example["messages"]:
            if msg.get("role") == "assistant":
                return msg["content"].strip()
    return None


def download_and_sample(dataset_id, config_name, split, n_samples, label):
    """Download a dataset split and sample n_samples prompts."""
    print(f"\n  [{label}] Loading {dataset_id} ({config_name}/{split})...", flush=True)
    try:
        if config_name:
            ds = load_dataset(dataset_id, config_name, split=split, trust_remote_code=True)
        else:
            ds = load_dataset(dataset_id, split=split, trust_remote_code=True)
    except Exception as e:
        print(f"    FAILED: {e}", flush=True)
        return []

    print(f"    Loaded {len(ds)} rows", flush=True)

    # Shuffle and take n_samples
    indices = list(range(len(ds)))
    random.shuffle(indices)

    prompts = []
    for idx in indices:
        if len(prompts) >= n_samples:
            break
        example = ds[idx]
        prompt = extract_prompt(example, dataset_id)
        if prompt and len(prompt) > 20 and len(prompt) < 10000:
            response = extract_response(example)
            entry = {"prompt": prompt, "source": label, "id": f"{label}_{len(prompts)}"}
            if response and len(response) > 10:
                entry["response"] = response
            prompts.append(entry)

    print(f"    Extracted {len(prompts)} prompts", flush=True)
    return prompts


def main():
    print("=" * 60)
    print(" DFlash v6 Dataset Preparation")
    print("=" * 60)
    print(f"  Output: {OUTPUT_DIR}")

    all_prompts = []

    # 1. Nemotron Post-Training v2 — code split (main z-lab source)
    prompts = download_and_sample(
        "nvidia/Nemotron-Post-Training-Dataset-v2", "SFT", "code",
        n_samples=25000, label="nemotron_code"
    )
    all_prompts.extend(prompts)

    # 2. Nemotron v2 — math split
    prompts = download_and_sample(
        "nvidia/Nemotron-Post-Training-Dataset-v2", "SFT", "math",
        n_samples=15000, label="nemotron_math"
    )
    all_prompts.extend(prompts)

    # 3. Nemotron v2 — chat split
    prompts = download_and_sample(
        "nvidia/Nemotron-Post-Training-Dataset-v2", "SFT", "chat",
        n_samples=15000, label="nemotron_chat"
    )
    all_prompts.extend(prompts)

    # 4. Nemotron v2 — stem split
    prompts = download_and_sample(
        "nvidia/Nemotron-Post-Training-Dataset-v2", "SFT", "stem",
        n_samples=10000, label="nemotron_stem"
    )
    all_prompts.extend(prompts)

    # 5. evol-codealpaca-v1 (z-lab used this, ~111K)
    prompts = download_and_sample(
        "theblackcat102/evol-codealpaca-v1", None, "train",
        n_samples=15000, label="evol_codealpaca"
    )
    all_prompts.extend(prompts)

    # 6. CodeFeedback-Filtered-Instruction (paper source, 157K)
    prompts = download_and_sample(
        "m-a-p/CodeFeedback-Filtered-Instruction", None, "train",
        n_samples=15000, label="codefeedback"
    )
    all_prompts.extend(prompts)

    # 7. Nemotron v2 — French (bonus diversity)
    prompts = download_and_sample(
        "nvidia/Nemotron-Post-Training-Dataset-v2", "SFT", "multilingual_fr",
        n_samples=5000, label="nemotron_french"
    )
    all_prompts.extend(prompts)

    # Shuffle and write
    random.shuffle(all_prompts)

    # Stats
    print(f"\n{'=' * 60}")
    print(f" Total: {len(all_prompts)} prompts")
    sources = {}
    has_response = 0
    for p in all_prompts:
        src = p["source"]
        sources[src] = sources.get(src, 0) + 1
        if "response" in p:
            has_response += 1

    for src, count in sorted(sources.items()):
        print(f"  {src}: {count}")
    print(f"  With pre-generated responses: {has_response}")
    print(f"  Need response generation: {len(all_prompts) - has_response}")

    # Write prompts JSONL
    output_file = OUTPUT_DIR / "prompts_100k.jsonl"
    with open(output_file, "w") as f:
        for p in all_prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"\n  Written to {output_file}")
    print(f"  Size: {output_file.stat().st_size / 1e6:.1f} MB")

    # Also write a version with just prompts that need response generation
    needs_response = [p for p in all_prompts if "response" not in p]
    if needs_response:
        nr_file = OUTPUT_DIR / "prompts_need_response.jsonl"
        with open(nr_file, "w") as f:
            for p in needs_response:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"  Prompts needing responses: {nr_file} ({len(needs_response)})")

    # Write a version ready for extraction (prompt+response pairs)
    has_resp = [p for p in all_prompts if "response" in p]
    if has_resp:
        ready_file = OUTPUT_DIR / "ready_for_extraction.jsonl"
        with open(ready_file, "w") as f:
            for p in has_resp:
                f.write(json.dumps({
                    "prompt": p["prompt"],
                    "response": p["response"],
                    "id": p["id"],
                }, ensure_ascii=False) + "\n")
        print(f"  Ready for extraction: {ready_file} ({len(has_resp)})")

    print(f"\n{'=' * 60}")
    print(f" Done!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
