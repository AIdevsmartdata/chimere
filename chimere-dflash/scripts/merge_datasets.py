#!/usr/bin/env python3
"""
merge_datasets.py — Merge all available datasets into ready_for_extraction format.

Output format: {"prompt": "...", "response": "...", "id": "source_NNN"}
Filters: skip samples > max_chars (proxy for 512 tokens), deduplicate by content hash.
"""

import json
import hashlib
import random
import sys
from pathlib import Path

MAX_CHARS = 1800  # ~512 tokens ≈ 1800 chars for English, conservative
MIN_CHARS = 50    # skip trivially short samples
MAX_PER_SOURCE = 50000  # cap per source for balance

def text_hash(text):
    return hashlib.md5(text[:500].encode()).hexdigest()

def extract_messages_text(messages):
    """Extract prompt and response from a messages list."""
    if isinstance(messages, str):
        messages = json.loads(messages)

    prompt_parts = []
    response = ""

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role in ("system", "user"):
            prompt_parts.append(content)
        elif role == "assistant":
            response = content  # take last assistant message

    prompt = "\n".join(prompt_parts)
    return prompt, response


def load_openr1_math(path, max_n):
    """openr1_math_220k: problem + solution fields, plus messages."""
    count = 0
    with open(path) as f:
        for line in f:
            if count >= max_n:
                break
            d = json.loads(line)
            # Use problem as prompt, solution as response
            prompt = d.get("problem", "")
            response = d.get("solution", "")
            if not prompt or not response:
                continue
            total = len(prompt) + len(response)
            if total < MIN_CHARS or total > MAX_CHARS:
                continue
            yield {"prompt": prompt, "response": response, "id": f"math_{count}"}
            count += 1


def load_codeforces(path, max_n):
    """codeforces_cots: prompt + generation fields."""
    count = 0
    with open(path) as f:
        for line in f:
            if count >= max_n:
                break
            d = json.loads(line)
            prompt = d.get("prompt", "")
            response = d.get("generation", "")
            if not prompt or not response:
                # Try messages format
                messages = d.get("messages")
                if messages:
                    prompt, response = extract_messages_text(messages)
            if not prompt or not response:
                continue
            total = len(prompt) + len(response)
            if total < MIN_CHARS or total > MAX_CHARS:
                continue
            yield {"prompt": prompt, "response": response, "id": f"codeforces_{count}"}
            count += 1


def load_mixture_of_thoughts(path, max_n):
    """mixture_of_thoughts_code: messages format."""
    count = 0
    with open(path) as f:
        for line in f:
            if count >= max_n:
                break
            d = json.loads(line)
            messages = d.get("messages")
            if not messages:
                continue
            prompt, response = extract_messages_text(messages)
            if not prompt or not response:
                continue
            total = len(prompt) + len(response)
            if total < MIN_CHARS or total > MAX_CHARS:
                continue
            yield {"prompt": prompt, "response": response, "id": f"mot_code_{count}"}
            count += 1


def load_kine(path, max_n, prefix="kine"):
    """synthetic_kine: messages format with system prompt."""
    count = 0
    with open(path) as f:
        for line in f:
            if count >= max_n:
                break
            d = json.loads(line)
            messages = d.get("messages")
            if not messages:
                continue
            prompt, response = extract_messages_text(messages)
            if not prompt or not response:
                continue
            total = len(prompt) + len(response)
            if total < MIN_CHARS or total > MAX_CHARS:
                continue
            yield {"prompt": prompt, "response": response, "id": f"{prefix}_{count}"}
            count += 1


def load_bfcl(path, max_n):
    """bfcl_v3_raw: question + function fields."""
    count = 0
    with open(path) as f:
        for line in f:
            if count >= max_n:
                break
            d = json.loads(line)
            prompt = d.get("question", "")
            response = d.get("function", "")
            if not prompt or not response:
                continue
            total = len(prompt) + len(response)
            if total < MIN_CHARS or total > MAX_CHARS:
                continue
            yield {"prompt": prompt, "response": response, "id": f"bfcl_{count}"}
            count += 1


def load_distillation(path, max_n, prefix="distill"):
    """Generic JSONL loader for distillation datasets (messages or prompt/response)."""
    count = 0
    with open(path) as f:
        for line in f:
            if count >= max_n:
                break
            d = json.loads(line)
            # Try messages format first
            messages = d.get("messages")
            if messages:
                prompt, response = extract_messages_text(messages)
            else:
                prompt = d.get("prompt", d.get("question", d.get("input", "")))
                response = d.get("response", d.get("answer", d.get("output", "")))

            if not prompt or not response:
                continue
            total = len(prompt) + len(response)
            if total < MIN_CHARS or total > MAX_CHARS:
                continue
            yield {"prompt": prompt, "response": response, "id": f"{prefix}_{count}"}
            count += 1


def load_existing(path):
    """Load existing ready_for_extraction.jsonl to avoid duplicates."""
    hashes = set()
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            h = text_hash(d.get("prompt", "") + d.get("response", ""))
            hashes.add(h)
    return hashes


def main():
    datasets_dir = Path.home() / ".chimere" / "datasets"
    distill_dir = Path.home() / ".chimere" / "workspaces" / "chimere" / "chimere-distillation" / "output"
    existing_file = Path.home() / "chimere-dflash" / "data" / "prompts_v6" / "ready_for_extraction.jsonl"
    output_file = Path.home() / "chimere-dflash" / "data" / "prompts_v6" / "merged_all.jsonl"

    # Load existing hashes
    print("Loading existing prompts for dedup...", flush=True)
    existing_hashes = load_existing(existing_file) if existing_file.exists() else set()
    print(f"  {len(existing_hashes)} existing samples", flush=True)

    sources = [
        ("openr1_math", lambda: load_openr1_math(datasets_dir / "openr1_math_220k.jsonl", MAX_PER_SOURCE)),
        ("codeforces", lambda: load_codeforces(datasets_dir / "codeforces_cots.jsonl", MAX_PER_SOURCE)),
        ("mot_code", lambda: load_mixture_of_thoughts(datasets_dir / "mixture_of_thoughts_code.jsonl", MAX_PER_SOURCE)),
        ("bfcl", lambda: load_bfcl(datasets_dir / "bfcl_v3_raw.jsonl", MAX_PER_SOURCE)),
        ("kine_v1", lambda: load_kine(datasets_dir / "synthetic-kine-v1" / "synthetic_kine_v1.jsonl", MAX_PER_SOURCE, "kinev1")),
        ("kine_v2", lambda: load_kine(datasets_dir / "synthetic-kine-v2" / "synthetic_kine_v2.jsonl", MAX_PER_SOURCE, "kinev2")),
    ]

    # Add distillation datasets if they exist
    if distill_dir.exists():
        for f in sorted(distill_dir.glob("*.jsonl")):
            name = f.stem
            sources.append((name, lambda p=f, n=name: load_distillation(p, 5000, n)))

    all_samples = []
    for name, loader in sources:
        print(f"Loading {name}...", end=" ", flush=True)
        count = 0
        dupes = 0
        for sample in loader():
            h = text_hash(sample["prompt"] + sample["response"])
            if h in existing_hashes:
                dupes += 1
                continue
            existing_hashes.add(h)
            all_samples.append(sample)
            count += 1
        print(f"{count} new ({dupes} dupes skipped)", flush=True)

    # Shuffle for diversity
    random.seed(42)
    random.shuffle(all_samples)

    # Write merged file
    with open(output_file, "w") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"\nTotal: {len(all_samples)} new samples written to {output_file}", flush=True)
    print(f"Combined with existing {len(existing_hashes) - len(all_samples)} = {len(existing_hashes)} total unique", flush=True)


if __name__ == "__main__":
    main()
