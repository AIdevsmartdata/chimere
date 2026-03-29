#!/usr/bin/env python3
"""
generate_eval_prompts.py — Generate 500 evaluation prompts for DFlash benchmarking.

Produces prompts that do NOT overlap with the training set (generate_prompts.py).
Uses a different seed (12345 vs 42) and completely different prompt templates.

Usage:
  python generate_eval_prompts.py
  python generate_eval_prompts.py --output data/eval_prompts.jsonl --count 500
"""

import argparse
import json
import random
from pathlib import Path

SEED_PROMPTS = {
    "code_python": [
        "Write a Python class implementing a thread pool executor with task priorities.",
        "Implement a Python A* pathfinding algorithm on a 2D grid.",
        "Create a Python function that validates and parses email addresses using regex.",
        "Write a Python script that implements a simple key-value store with persistence.",
        "Implement a bloom filter in Python with configurable false positive rate.",
        "Create a Python generator that yields prime numbers using the Sieve of Eratosthenes.",
        "Write a Python function that compresses text using Huffman encoding.",
        "Implement a simple Python ORM that maps classes to SQLite tables.",
        "Create a Python async websocket server that broadcasts messages to all clients.",
        "Write a Python function that performs topological sort on a directed acyclic graph.",
    ],
    "code_rust": [
        "Implement a Rust channel-based actor system with typed messages.",
        "Write a Rust macro that generates builder pattern code for structs.",
        "Create a Rust iterator adapter that groups consecutive equal elements.",
        "Implement a Rust lock-free stack using atomic operations.",
        "Write a Rust function that deserializes JSON into a dynamic value type.",
        "Create a Rust trait object dispatcher for plugin architectures.",
    ],
    "code_js": [
        "Write a JavaScript function that deep-clones objects handling circular references.",
        "Create a TypeScript generic retry wrapper with exponential backoff and jitter.",
        "Implement a JavaScript virtual DOM diffing algorithm.",
        "Write a Node.js stream transform that parses newline-delimited JSON.",
        "Create a JavaScript Proxy-based reactive state management library.",
    ],
    "math": [
        "Solve the optimization problem: minimize f(x,y) = x^2 + y^2 subject to x + y = 10.",
        "Prove that every continuous function on [0,1] is uniformly continuous.",
        "Calculate the determinant of a 4x4 matrix using cofactor expansion.",
        "Find the general solution of the system: dx/dt = 2x - y, dy/dt = x + 3y.",
        "Prove that the set of rational numbers is countable using Cantor's diagonal argument.",
        "Evaluate the integral of sin(x)/x from 0 to infinity.",
    ],
    "french": [
        "Explique le principe de fonctionnement d'un compilateur JIT et ses avantages par rapport a la compilation AOT.",
        "Decris les differents types d'attaques par injection SQL et les methodes de prevention.",
        "Redige un guide sur la mise en place d'un pipeline CI/CD avec GitLab.",
        "Explique le theoreme CAP et ses implications pour les systemes distribues.",
        "Decris l'architecture d'un moteur de recherche full-text comme Elasticsearch.",
        "Explique le fonctionnement des protocoles de consensus Raft et Paxos.",
    ],
    "english": [
        "Explain how mRNA vaccines work and the role of lipid nanoparticles in delivery.",
        "Describe the architecture of a GPU and how CUDA parallelism works.",
        "Explain the difference between symmetric and asymmetric encryption with examples.",
        "Describe how neural network pruning and quantization reduce model size.",
        "Explain the CAP theorem and its implications for distributed databases.",
        "Describe how garbage collection works in Go compared to Rust's ownership model.",
    ],
    "creative": [
        "Write a technical thriller short story about a zero-day vulnerability.",
        "Compose a sonnet about debugging code at midnight.",
        "Write a dialogue between a CPU and GPU arguing about workload distribution.",
        "Create a blog post comparing three approaches to distributed consensus.",
    ],
}

VARIATION_PREFIXES = [
    "",
    "You are an expert software architect. ",
    "You are a university professor. ",
    "Tu es un ingenieur senior specialise en systemes distribues. ",
    "Answer with detailed explanations: ",
    "Provide a step-by-step solution: ",
    "Be concise but rigorous: ",
    "You are a security researcher. ",
]


def main():
    parser = argparse.ArgumentParser(description="Generate eval prompts for DFlash")
    parser.add_argument("--output", type=str, default="data/eval_prompts.jsonl")
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    random.seed(args.seed)

    # Collect base prompts
    all_prompts = []
    for domain, prompts in SEED_PROMPTS.items():
        for i, text in enumerate(prompts):
            all_prompts.append({"text": text, "id": f"eval_{domain}_{i}", "domain": domain})

    # Expand to target count with variation prefixes
    base_prompts = list(all_prompts)
    while len(all_prompts) < args.count:
        base = random.choice(base_prompts)
        prefix = random.choice(VARIATION_PREFIXES)
        all_prompts.append({
            "text": prefix + base["text"],
            "id": f"eval_{base['domain']}_{len(all_prompts)}",
            "domain": base["domain"],
        })

    all_prompts = all_prompts[:args.count]
    random.shuffle(all_prompts)

    # Write JSONL
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for prompt in all_prompts:
            entry = {"text": prompt["text"], "id": prompt["id"]}
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Stats
    domains = {}
    for p in all_prompts:
        domains[p["domain"]] = domains.get(p["domain"], 0) + 1
    print(f"Generated {len(all_prompts)} eval prompts -> {args.output}")
    print("Domain distribution:")
    for d, c in sorted(domains.items()):
        print(f"  {d}: {c}")


if __name__ == "__main__":
    main()
