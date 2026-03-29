#!/usr/bin/env python3
"""
generate_prompts.py — Generate diverse prompts for DFlash feature extraction.

Generates a JSONL file of prompts covering multiple domains:
  - Code generation (Python, Rust, JavaScript, SQL)
  - Math and reasoning
  - French technical text
  - English general knowledge
  - Tool calling / structured output
  - Creative writing

These prompts are fed to extract_hidden_states to capture Qwen3.5's internal
representations for training the DFlash drafter.

Usage:
  python generate_prompts.py --output data/prompts.jsonl --count 500
  python generate_prompts.py --output data/prompts.jsonl --count 500 --generate-responses
"""

import argparse
import json
import random
from pathlib import Path

# Seed prompts covering diverse domains
SEED_PROMPTS = {
    "code_python": [
        "Write a Python function that implements binary search on a sorted list.",
        "Create a Python class for a thread-safe LRU cache with TTL support.",
        "Implement a Python generator that yields Fibonacci numbers lazily.",
        "Write a Python script that parses command-line arguments using argparse and processes CSV files.",
        "Create a Python decorator that retries a function on exception with exponential backoff.",
        "Implement a simple HTTP server in Python that serves static files and handles POST requests.",
        "Write a Python function to merge two sorted linked lists into one sorted list.",
        "Create a Python context manager for temporary directory creation and cleanup.",
        "Write a Python async function that fetches multiple URLs concurrently with aiohttp.",
        "Implement a trie data structure in Python with insert, search, and prefix matching.",
    ],
    "code_rust": [
        "Write a Rust function that implements a concurrent hash map using RwLock.",
        "Create a Rust struct that represents a binary tree with methods for insertion and traversal.",
        "Implement error handling in Rust using custom error types and the thiserror crate.",
        "Write a Rust function that reads a file line by line and counts word frequencies.",
        "Create a Rust trait for serializable data structures with default implementations.",
        "Write a Rust program that uses tokio for async TCP server handling.",
        "Implement a simple memory allocator in Rust using a free list.",
        "Create a Rust enum for a calculator AST with evaluation method.",
    ],
    "code_js": [
        "Write a JavaScript function that debounces user input with configurable delay.",
        "Create a React component that implements infinite scrolling with virtualization.",
        "Implement a JavaScript Promise.all polyfill that handles both success and failure.",
        "Write a Node.js middleware for rate limiting API requests using a sliding window.",
        "Create a JavaScript class for an event emitter with once, on, and off methods.",
    ],
    "code_sql": [
        "Write a SQL query to find the top 5 customers by total order value with running totals.",
        "Create a SQL stored procedure that handles inventory updates with transaction safety.",
        "Write a SQL query using window functions to calculate moving averages over 7-day periods.",
        "Design a SQL schema for a multi-tenant SaaS application with proper indexing strategy.",
    ],
    "math_reasoning": [
        "Solve step by step: If f(x) = x^3 - 3x^2 + 2x, find all critical points and classify them.",
        "Prove that the square root of 2 is irrational using proof by contradiction.",
        "Calculate the eigenvalues and eigenvectors of the matrix [[3, 1], [1, 3]].",
        "Solve the recurrence relation T(n) = 2T(n/2) + n using the Master Theorem.",
        "Find the Taylor series expansion of e^x around x=0 up to the 5th order term.",
        "Solve the differential equation dy/dx = y*sin(x) with y(0) = 1.",
        "Prove by induction that the sum of the first n odd numbers equals n squared.",
        "Calculate the volume of a solid of revolution when y = sqrt(x) is rotated around the x-axis from 0 to 4.",
    ],
    "french_technical": [
        "Explique le fonctionnement d'un reseau de neurones convolutif (CNN) pour la classification d'images.",
        "Decris l'architecture d'un systeme de microservices avec Docker et Kubernetes.",
        "Redige une documentation technique pour une API REST de gestion de bibliotheque.",
        "Explique les differences entre les bases de donnees relationnelles et NoSQL avec des exemples concrets.",
        "Decris le protocole HTTPS et le processus de handshake TLS etape par etape.",
        "Ecris un guide technique sur la configuration d'un serveur Nginx comme reverse proxy.",
        "Explique le fonctionnement de la memoire virtuelle et de la pagination dans un systeme d'exploitation.",
        "Redige une analyse comparative des frameworks web Python : Django, Flask et FastAPI.",
        "Decris les principes SOLID en programmation orientee objet avec des exemples en Python.",
        "Explique le fonctionnement du garbage collector en Java et ses differentes strategies.",
    ],
    "english_knowledge": [
        "Explain how CRISPR-Cas9 gene editing works and its potential medical applications.",
        "Describe the architecture of a modern CPU including pipelining, caching, and branch prediction.",
        "Explain the difference between classical and quantum computing with practical examples.",
        "Describe how blockchain consensus mechanisms work, comparing Proof of Work and Proof of Stake.",
        "Explain the principles behind reinforcement learning and how AlphaGo was trained.",
        "Describe the process of protein folding and why it's computationally challenging.",
        "Explain how modern recommendation systems work, including collaborative and content-based filtering.",
        "Describe the architecture and training process of large language models like GPT and Claude.",
    ],
    "tool_calling": [
        'You have access to a function called search(query: str) -> list[dict]. Use it to find recent papers about attention mechanisms. Format your response as JSON.',
        'Given the function calculate(expression: str) -> float, compute the compound interest on $10000 at 5% for 10 years.',
        'You have a database with tables: users(id, name, email), orders(id, user_id, total, date). Write a query to find users who spent more than $1000 last month.',
        'Using the API endpoint POST /api/tasks with body {"title": str, "priority": int, "due_date": str}, create 3 tasks for a software release.',
        'Extract structured data from this text: "John Smith, age 35, works at Acme Corp as Senior Engineer since 2019. Contact: john@acme.com"',
    ],
    "creative": [
        "Write a short science fiction story about an AI that discovers it can dream.",
        "Compose a poem about the intersection of mathematics and nature.",
        "Write a dialogue between two programmers debugging a mysterious production issue at 3 AM.",
        "Create a technical blog post introduction about the future of edge computing.",
        "Write a persuasive essay about why open-source software is essential for innovation.",
    ],
}


def generate_meta_prompts():
    """Generate prompts that ask the model to generate more prompts (self-play)."""
    templates = [
        "Generate 5 diverse Python coding challenges suitable for a technical interview, with varying difficulty levels.",
        "Create 5 mathematical word problems that require multi-step reasoning to solve.",
        "Write 5 different system design questions for a software engineering interview.",
        "Generate 5 creative writing prompts that blend technology and philosophy.",
        "Create 5 questions about machine learning that test deep understanding rather than surface knowledge.",
    ]
    return [{"text": t, "id": f"meta_{i}", "domain": "meta"} for i, t in enumerate(templates)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="data/prompts.jsonl")
    parser.add_argument("--count", type=int, default=500,
                        help="Target number of prompts (will repeat/shuffle if needed)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generate-responses", action="store_true",
                        help="Also generate responses via Qwen3.5 API (port 8081)")
    parser.add_argument("--api-url", type=str, default="http://localhost:8081/v1/chat/completions")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Max tokens for generated responses")
    args = parser.parse_args()

    random.seed(args.seed)

    # Collect all prompts
    all_prompts = []
    idx = 0
    for domain, prompts in SEED_PROMPTS.items():
        for text in prompts:
            all_prompts.append({
                "text": text,
                "id": f"{domain}_{idx}",
                "domain": domain,
            })
            idx += 1

    # Add meta prompts
    all_prompts.extend(generate_meta_prompts())

    # Expand to target count by adding variations
    if len(all_prompts) < args.count:
        system_prefixes = [
            "",
            "You are a helpful coding assistant. ",
            "You are an expert mathematician. ",
            "Tu es un assistant technique expert en informatique. ",
            "You are a senior software engineer reviewing code. ",
            "Explain clearly and concisely: ",
            "Think step by step and solve: ",
            "Be precise and thorough: ",
        ]

        expanded = list(all_prompts)
        while len(expanded) < args.count:
            base = random.choice(all_prompts)
            prefix = random.choice(system_prefixes)
            expanded.append({
                "text": prefix + base["text"],
                "id": f"{base['id']}_var{len(expanded)}",
                "domain": base["domain"],
            })

        all_prompts = expanded[:args.count]

    random.shuffle(all_prompts)

    # Write JSONL — stream mode: write each prompt as it's generated
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: load already-written IDs
    existing_ids = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    existing_ids.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    pass
        if existing_ids:
            print(f"  Resuming: {len(existing_ids)} prompts already done")

    # Generate responses if requested
    if args.generate_responses:
        try:
            import requests
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry
        except ImportError:
            print("ERROR: requests library needed for --generate-responses")
            print("  pip install requests")
            return

        # Robust session with retries
        session = requests.Session()
        retry = Retry(total=2, backoff_factor=1, status_forcelist=[502, 503, 504])
        session.mount("http://", HTTPAdapter(max_retries=retry))

        print(f"Generating responses via {args.api_url}...")
        n_success = 0
        n_failed = 0

        with open(output_path, "a") as f:
            for i, prompt in enumerate(all_prompts):
                if prompt["id"] in existing_ids:
                    continue

                if i % 10 == 0:
                    print(f"  [{i}/{len(all_prompts)}] (ok={n_success}, fail={n_failed})...",
                          flush=True)
                try:
                    resp = session.post(args.api_url, json={
                        "model": "qwen3.5",
                        "messages": [{"role": "user", "content": prompt["text"]}],
                        "max_tokens": args.max_tokens,
                        "temperature": 0.7,
                        "chat_template_kwargs": {"enable_thinking": False},
                    }, timeout=60)
                    if resp.status_code == 200:
                        data = resp.json()
                        msg = data["choices"][0]["message"]
                        response_text = msg.get("content") or msg.get("reasoning_content") or ""
                        if response_text:
                            text = prompt["text"] + "\n" + response_text
                        else:
                            text = prompt["text"]
                        n_success += 1
                    else:
                        text = prompt["text"]
                        n_failed += 1
                except Exception as e:
                    print(f"  [WARN] Failed for {prompt['id']}: {e}", flush=True)
                    text = prompt["text"]
                    n_failed += 1

                # Write immediately
                entry = {"text": text, "id": prompt["id"]}
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                f.flush()

        print(f"\n  Response generation done: {n_success} OK, {n_failed} failed")
    else:
        # No response generation — just write prompts
        with open(output_path, "a") as f:
            for prompt in all_prompts:
                if prompt["id"] in existing_ids:
                    continue
                entry = {"text": prompt["text"], "id": prompt["id"]}
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    total = len(existing_ids) + len(all_prompts) - len(existing_ids)
    print(f"\nGenerated {total} prompts -> {args.output}")
    domains = {}
    for p in all_prompts:
        d = p["domain"]
        domains[d] = domains.get(d, 0) + 1
    print("Domain distribution:")
    for d, c in sorted(domains.items()):
        print(f"  {d}: {c}")


if __name__ == "__main__":
    main()
