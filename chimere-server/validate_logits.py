#!/usr/bin/env python3
"""Validate chimere-deltanet logits against llama.cpp reference.

Usage:
  1. Start llama.cpp with the 27B Opus MTP GGUF on port 8091
  2. Run chimere-deltanet forward on the same prompt
  3. Compare top-5 logits

Step 1: Generate reference
  model-swap opus  # or manually start on port 8091
  python3 validate_logits.py --generate-reference

Step 2: Compare (after chimere forward is working)
  python3 validate_logits.py --compare chimere_logits.json
"""

import json
import sys
import argparse
import urllib.request

LLAMA_URL = "http://127.0.0.1:8091/v1/chat/completions"
PROMPT = "Hello"
REF_FILE = "reference_logits.json"

def generate_reference():
    payload = json.dumps({
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 1,
        "logprobs": True,
        "top_logprobs": 20,
        "temperature": 0.0,
    }).encode()

    req = urllib.request.Request(LLAMA_URL, data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        r = json.loads(resp.read())

    lp = r["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    ref = {t["token"]: t["logprob"] for t in lp}

    with open(REF_FILE, "w") as f:
        json.dump({"prompt": PROMPT, "top_logprobs": ref}, f, indent=2)

    print(f"Reference saved to {REF_FILE}")
    print(f"Top-5 tokens:")
    for i, (tok, lp) in enumerate(sorted(ref.items(), key=lambda x: -x[1])[:5]):
        print(f"  {i+1}. {tok:20s} {lp:.4f}")

def compare(chimere_file):
    with open(REF_FILE) as f:
        ref = json.load(f)
    with open(chimere_file) as f:
        chimere = json.load(f)

    ref_lp = ref["top_logprobs"]
    chi_lp = chimere.get("top_logprobs", {})

    print("Token                llama.cpp    chimere      delta")
    print("-" * 60)

    matches = 0
    for tok in sorted(ref_lp, key=lambda t: -ref_lp[t])[:10]:
        r = ref_lp[tok]
        c = chi_lp.get(tok, float("-inf"))
        delta = abs(r - c)
        ok = "OK" if delta < 0.1 else "MISMATCH"
        if delta < 0.1:
            matches += 1
        print(f"{tok:20s} {r:+8.4f}     {c:+8.4f}     {delta:.4f}  {ok}")

    print(f"\nMatch rate: {matches}/10 ({matches*10}%)")
    if matches >= 8:
        print("PASS — logits match within tolerance")
    else:
        print("FAIL — logits diverge significantly")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate-reference", action="store_true")
    parser.add_argument("--compare", type=str)
    args = parser.parse_args()

    if args.generate_reference:
        generate_reference()
    elif args.compare:
        compare(args.compare)
    else:
        parser.print_help()
