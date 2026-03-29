#!/usr/bin/env python3
"""Extract MoE gate weights from Qwen3.5-35B-A3B GGUF for expert prefetch head.

Extracts `blk.{L}.ffn_gate_inp.weight` for CPU-offloaded layers (20-39).
These are the router/gating matrices stored in GGUF as [2048, 256] (input_dim x n_experts),
transposed to nn.Linear convention [256, 2048] (n_experts x n_embd) in the output.

Output: data/qwen_gate_weights_20_39.pt
  Dict[int, Tensor[256, 2048]] — transposed to match nn.Linear convention.

Usage:
  python scripts/extract_gate_weights.py
  python scripts/extract_gate_weights.py --model /path/to/model.gguf --output data/out.pt
  python scripts/extract_gate_weights.py --layers 20,21,22,30
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

try:
    from gguf import GGUFReader
except ImportError:
    print("ERROR: 'gguf' package not found. Install with: pip install gguf", file=sys.stderr)
    sys.exit(1)

DEFAULT_GGUF = str(
    Path.home() / ".chimere/models/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-UD-Q5_K_XL.gguf"
)
DEFAULT_LAYERS = "20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39"
DEFAULT_OUTPUT = "data/qwen_gate_weights_20_39.pt"

# Expected dimensions for Qwen3.5-35B-A3B
N_EXPERTS = 256
N_EMBD = 2048


def parse_layers(layers_str: str) -> list[int]:
    """Parse layers from comma-separated list or dash range (e.g. '20,21,22' or '20-39')."""
    layers_str = layers_str.strip()
    if "," in layers_str:
        return [int(x.strip()) for x in layers_str.split(",")]
    if "-" in layers_str:
        parts = layers_str.split("-")
        if len(parts) == 2:
            start, end = int(parts[0]), int(parts[1])
            return list(range(start, end + 1))
    # Single layer
    return [int(layers_str)]


def extract_gate_weights(gguf_path: str, layers: list[int], output_path: str) -> None:
    gguf_path = str(Path(gguf_path).expanduser())
    if not Path(gguf_path).exists():
        print(f"ERROR: GGUF file not found: {gguf_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Reading GGUF: {gguf_path}")
    print(f"Target layers: {layers}")
    print()

    reader = GGUFReader(gguf_path)

    # Build lookup: tensor_name -> tensor object for fast access
    target_names = {f"blk.{layer_id}.ffn_gate_inp.weight": layer_id for layer_id in layers}
    found_names: set[str] = set()

    gate_weights: dict[int, torch.Tensor] = {}

    for tensor in reader.tensors:
        tname = tensor.name
        if tname not in target_names:
            continue

        layer_id = target_names[tname]
        found_names.add(tname)

        # GGUF stores gate_inp.weight as [n_embd, n_experts] = [2048, 256].
        # Gate weights for small router tensors are typically stored as F32 or F16
        # (not block-quantized like the expert ffn weights).
        raw = tensor.data  # numpy memmap, shape as stored in GGUF

        # Flatten and cast to float32 in case it's F16
        arr = np.array(raw, dtype=np.float32)

        # Reshape to [n_embd, n_experts] based on GGUF shape field.
        # GGUF shape is stored in reverse order (column-major): shape[0]=n_experts, shape[1]=n_embd.
        # After reshape(tensor.shape) we get [n_experts, n_embd] = [256, 2048].
        # We want final shape [256, 2048] (n_experts x n_embd) matching nn.Linear weight layout.
        gguf_shape = list(tensor.shape)  # GGUF reverses dims vs numpy convention

        if len(gguf_shape) == 2:
            # GGUF shape: [n_cols, n_rows] = [2048, 256] in GGUF convention
            # numpy reshape gives [2048, 256] unless we account for GGUF column-major ordering.
            # In practice GGUFReader exposes shape already corrected — verify:
            arr_2d = arr.reshape(gguf_shape)
            if arr_2d.shape == (N_EMBD, N_EXPERTS):
                # Stored as [n_embd, n_experts], transpose to [n_experts, n_embd]
                w = torch.from_numpy(arr_2d.T.copy())
            elif arr_2d.shape == (N_EXPERTS, N_EMBD):
                # Already [n_experts, n_embd]
                w = torch.from_numpy(arr_2d.copy())
            else:
                print(
                    f"  WARNING: {tname} has unexpected shape {arr_2d.shape}, "
                    f"expected ({N_EMBD}, {N_EXPERTS}) or ({N_EXPERTS}, {N_EMBD}). "
                    f"Storing as-is reshaped to ({N_EXPERTS}, {N_EMBD}).",
                    file=sys.stderr,
                )
                w = torch.from_numpy(arr.reshape(N_EXPERTS, N_EMBD).copy())
        else:
            # Unexpected rank — flatten and reshape
            print(
                f"  WARNING: {tname} has {len(gguf_shape)}D shape {gguf_shape}, "
                f"expected 2D. Forcing reshape to ({N_EXPERTS}, {N_EMBD}).",
                file=sys.stderr,
            )
            w = torch.from_numpy(arr.reshape(N_EXPERTS, N_EMBD).copy())

        gate_weights[layer_id] = w

        # Per-layer shape verification
        shape_ok = list(w.shape) == [N_EXPERTS, N_EMBD]
        status = "OK" if shape_ok else "SHAPE MISMATCH"
        print(
            f"  Layer {layer_id:2d}  {tname}  "
            f"GGUF shape={gguf_shape}  ->  weight shape={list(w.shape)}  "
            f"dtype={w.dtype}  [{status}]"
        )

    print()

    # Report missing layers
    missing_layers = sorted(set(layers) - set(gate_weights.keys()))
    if missing_layers:
        missing_names = [f"blk.{l}.ffn_gate_inp.weight" for l in missing_layers]
        print(f"WARNING: {len(missing_layers)} tensor(s) not found in GGUF:", file=sys.stderr)
        for n in missing_names:
            print(f"  - {n}", file=sys.stderr)

    if not gate_weights:
        print("ERROR: No gate weights extracted. Aborting.", file=sys.stderr)
        sys.exit(1)

    # Summary stats
    n_extracted = len(gate_weights)
    total_params = sum(w.numel() for w in gate_weights.values())
    total_mb = total_params * 4 / 1024**2  # float32 = 4 bytes

    print(f"Extracted {n_extracted}/{len(layers)} gate weight matrices")
    print(f"Shape per layer: [{N_EXPERTS}, {N_EMBD}] (n_experts x n_embd)")
    print(f"Total parameters: {total_params:,}  ({total_mb:.2f} MB float32)")

    # Save
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(gate_weights, str(out))
    print(f"\nSaved to: {out.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract MoE gate weights (ffn_gate_inp) from Qwen3.5-35B-A3B GGUF.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", "-m",
        default=DEFAULT_GGUF,
        help="Path to the GGUF model file.",
    )
    parser.add_argument(
        "--layers", "-l",
        default=DEFAULT_LAYERS,
        help=(
            "Layers to extract, as a comma-separated list (e.g. '20,21,22') "
            "or a dash range (e.g. '20-39')."
        ),
    )
    parser.add_argument(
        "--output", "-o",
        default=DEFAULT_OUTPUT,
        help="Output .pt file path. Saved as Dict[int, Tensor[256, 2048]].",
    )
    args = parser.parse_args()

    layers = parse_layers(args.layers)
    if not layers:
        print("ERROR: No layers specified.", file=sys.stderr)
        sys.exit(1)

    extract_gate_weights(args.model, layers, args.output)


if __name__ == "__main__":
    main()
