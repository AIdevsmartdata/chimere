"""
Online experience buffer for on-policy speculative decoding training.

Captures (hidden_states, draft_tokens, target_tokens, accepted_mask) from
each spec decode cycle and stores them in the same format as features_fullseq/
for direct reuse with the existing training pipeline.

Buffer is circular: oldest samples are overwritten when capacity is reached.
"""
import json
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


class OnlineBuffer:
    """Circular buffer of spec decode experiences on disk.

    Each sample is stored as a directory with:
      - context_hidden.bin  [n_layers, ctx_len, hidden_dim] float16
      - tokens.bin          [seq_len] int32
      - metadata.json       {n_positions, layers, hidden_dim, ...}
      - online_meta.json    {draft_tokens, accepted_mask, n_accepted, timestamp}

    Compatible with DFlashFullSeqDirDataset for training.
    """

    def __init__(
        self,
        buffer_dir: str,
        capacity: int = 10000,
        layers: List[int] = None,
        hidden_dim: int = 2048,
    ):
        self.buffer_dir = Path(buffer_dir)
        self.buffer_dir.mkdir(parents=True, exist_ok=True)
        self.capacity = capacity
        self.layers = layers or [1, 10, 19, 28, 37]
        self.hidden_dim = hidden_dim

        # Track write position
        self._state_path = self.buffer_dir / "_buffer_state.json"
        self._load_state()

    def _load_state(self):
        if self._state_path.exists():
            with open(self._state_path) as f:
                state = json.load(f)
            self._write_idx = state.get("write_idx", 0)
            self._total_written = state.get("total_written", 0)
        else:
            self._write_idx = 0
            self._total_written = 0

    def _save_state(self):
        with open(self._state_path, "w") as f:
            json.dump({
                "write_idx": self._write_idx,
                "total_written": self._total_written,
                "capacity": self.capacity,
            }, f)

    @property
    def size(self) -> int:
        """Number of valid samples in buffer."""
        return min(self._total_written, self.capacity)

    def store(
        self,
        hidden_states: np.ndarray,
        tokens: np.ndarray,
        draft_tokens: List[int],
        target_tokens: List[int],
        accepted_mask: List[bool],
        anchor_pos: int,
        source: str = "online",
    ) -> Path:
        """Store one spec decode experience.

        Args:
            hidden_states: [n_layers, seq_len, hidden_dim] float32 from target eval
            tokens: [seq_len] int32 — full token sequence
            draft_tokens: list of K drafted token IDs
            target_tokens: list of K target token IDs (ground truth)
            accepted_mask: list of K bools (True = draft matched target)
            anchor_pos: int — position in sequence where drafting started
            source: str — identifier for this experience source

        Returns:
            Path to the stored sample directory
        """
        sample_id = self._write_idx % self.capacity
        sample_dir = self.buffer_dir / f"sample_{sample_id:06d}"

        # Overwrite if exists (circular)
        if sample_dir.exists():
            shutil.rmtree(sample_dir)
        sample_dir.mkdir()

        # Convert to float16 for storage (matches offline extraction format)
        h16 = hidden_states.astype(np.float16)
        h16.tofile(sample_dir / "context_hidden.bin")

        tokens_i32 = tokens.astype(np.int32)
        tokens_i32.tofile(sample_dir / "tokens.bin")

        n_accepted = sum(accepted_mask)

        # Standard metadata (compatible with DFlashFullSeqDirDataset)
        metadata = {
            "anchor_pos": anchor_pos,
            "seq_len": len(tokens),
            "block_size": len(draft_tokens) + 1,  # includes anchor
            "ctx_len": 0,
            "n_positions": len(tokens),
            "layers": self.layers,
            "hidden_dim": self.hidden_dim,
            "dtype": "float16",
            "mode": "full_seq",
            "source_id": f"{source}_{self._total_written}",
        }
        with open(sample_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Online-specific metadata (for analysis, not needed by training)
        online_meta = {
            "draft_tokens": draft_tokens,
            "target_tokens": target_tokens,
            "accepted_mask": accepted_mask,
            "n_accepted": n_accepted,
            "n_drafted": len(draft_tokens),
            "timestamp": time.time(),
            "source": source,
        }
        with open(sample_dir / "online_meta.json", "w") as f:
            json.dump(online_meta, f, indent=2)

        self._write_idx += 1
        self._total_written += 1
        # Save state every 100 writes (not every single one)
        if self._total_written % 100 == 0:
            self._save_state()

        return sample_dir

    def get_recent(self, n: int = 500) -> List[Path]:
        """Get the N most recent sample directories."""
        if self._total_written == 0:
            return []

        n = min(n, self.size)
        dirs = []
        for i in range(n):
            idx = (self._write_idx - 1 - i) % self.capacity
            d = self.buffer_dir / f"sample_{idx:06d}"
            if d.exists() and (d / "context_hidden.bin").exists():
                dirs.append(d)
        return dirs

    def flush(self):
        """Force save buffer state to disk."""
        self._save_state()

    def get_all_dirs(self) -> List[Path]:
        """Get all valid sample directories (for training)."""
        return sorted([
            d for d in self.buffer_dir.iterdir()
            if d.is_dir() and (d / "context_hidden.bin").exists()
        ])

    def stats(self) -> Dict:
        """Buffer statistics."""
        dirs = self.get_all_dirs()
        n_accepted_total = 0
        n_drafted_total = 0
        for d in dirs:
            meta_path = d / "online_meta.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    om = json.load(f)
                n_accepted_total += om.get("n_accepted", 0)
                n_drafted_total += om.get("n_drafted", 0)

        return {
            "size": len(dirs),
            "capacity": self.capacity,
            "total_written": self._total_written,
            "tau": n_accepted_total / max(1, n_drafted_total),
            "n_accepted": n_accepted_total,
            "n_drafted": n_drafted_total,
        }
