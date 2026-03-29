"""
Online speculative decoding with experience capture.

Uses TargetDaemon (C++ binary protocol) for fast hidden state capture,
DFlash drafter for block generation, and OnlineBuffer for storing
experiences for on-policy fine-tuning.

Replaces the HTTP-based spec_decode.py with a fully integrated pipeline.
"""
import time
from typing import List, Optional, Tuple

import numpy as np
import torch

from .config_v7 import DFlashV7Config
from .online_buffer import OnlineBuffer
from .target_daemon import TargetDaemon


class OnlineSpecDecoder:
    """Speculative decoder with on-policy experience capture.

    Flow per generation step:
      1. Target prefill → hidden_states (captured for free)
      2. Drafter generates K draft tokens from hidden_states
      3. Target verifies draft tokens → accept/reject
      4. Store (hidden_states, draft, target, mask) in buffer
      5. Return accepted tokens to user
    """

    def __init__(
        self,
        drafter: torch.nn.Module,
        target: TargetDaemon,
        config: DFlashV7Config,
        buffer: Optional[OnlineBuffer] = None,
        device: str = "cuda",
    ):
        self.drafter = drafter
        self.target = target
        self.config = config
        self.buffer = buffer
        self.device = torch.device(device)
        self.K = config.block_size
        self.n_layers = config.num_feature_layers
        self.H = config.target_hidden_size

        # Stats
        self.total_accepted = 0
        self.total_drafted = 0
        self.total_calls = 0

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        capture: bool = True,
    ) -> Tuple[str, dict]:
        """Generate text with speculative decoding + experience capture.

        Args:
            prompt: input text
            max_tokens: maximum tokens to generate
            temperature: sampling temperature (0 = greedy)
            capture: whether to store experiences in buffer

        Returns:
            (generated_text, stats_dict)
        """
        # Tokenize
        tokens = self.target.tokenize(prompt)
        generated = []
        stats = {"n_calls": 0, "n_accepted": 0, "n_drafted": 0, "draft_ms": 0}

        # Initial prefill: get hidden states for the prompt
        hidden_all, argmax = self.target.eval_full(tokens)
        # hidden_all: [n_layers, seq_len, hidden_dim] float32

        while len(generated) < max_tokens:
            seq_len = len(tokens) + len(generated)
            if seq_len < 2:
                break

            # Prepare context for drafter: last max_ctx_len positions
            ctx_end = hidden_all.shape[1]
            ctx_start = max(0, ctx_end - self.config.max_ctx_len)
            ctx_len = ctx_end - ctx_start

            hidden_list = [
                torch.from_numpy(hidden_all[i, ctx_start:ctx_end])
                .unsqueeze(0).to(self.device)
                for i in range(self.n_layers)
            ]
            ctx_lengths = torch.tensor([ctx_len], device=self.device, dtype=torch.long)
            anchor_pos = ctx_end - 1
            anchor_positions = torch.tensor([anchor_pos], device=self.device, dtype=torch.long)

            # Last token as anchor
            all_tokens = tokens + generated
            anchor_token_id = all_tokens[-1]

            # Draft K tokens
            t0 = time.time()
            draft_ids, _, expert_preds = self.drafter.generate_block(
                hidden_list,
                context_lengths=ctx_lengths,
                temperature=temperature,
                anchor_token_id=anchor_token_id,
                anchor_positions=anchor_positions,
            )
            draft_ms = (time.time() - t0) * 1000
            stats["draft_ms"] += draft_ms

            draft_tokens = draft_ids[0].cpu().tolist()  # [K-1]

            # Target verifies: eval draft tokens incrementally
            target_hidden, target_argmax = self.target.eval_incr(draft_tokens)
            target_tokens = target_argmax.tolist()  # [K-1]

            # Accept sequentially until mismatch
            accepted_mask = []
            n_accepted = 0
            for j in range(len(draft_tokens)):
                if j < len(target_tokens) and draft_tokens[j] == target_tokens[j]:
                    accepted_mask.append(True)
                    n_accepted += 1
                else:
                    accepted_mask.append(False)
                    break
            # Pad remaining positions as rejected
            while len(accepted_mask) < len(draft_tokens):
                accepted_mask.append(False)

            # Add accepted tokens + correction token to output
            for j in range(n_accepted):
                generated.append(draft_tokens[j])
            if n_accepted < len(draft_tokens) and n_accepted < len(target_tokens):
                # Add target's correction token
                generated.append(target_tokens[n_accepted])
            elif n_accepted == len(draft_tokens) and len(target_tokens) > n_accepted:
                # All accepted → bonus token
                generated.append(target_tokens[-1])

            # Update hidden states: append target hidden for accepted+correction
            # The eval_incr already computed these
            n_new = n_accepted + 1
            if n_new <= target_hidden.shape[1]:
                hidden_all = np.concatenate(
                    [hidden_all, target_hidden[:, :n_new, :]], axis=1
                )
            else:
                hidden_all = np.concatenate(
                    [hidden_all, target_hidden], axis=1
                )

            # Trim KV cache in target to match
            self.target.trim_kv(len(tokens) + len(generated))

            # Capture experience
            if capture and self.buffer is not None:
                full_tokens = np.array(tokens + generated, dtype=np.int32)
                self.buffer.store(
                    hidden_states=hidden_all,
                    tokens=full_tokens,
                    draft_tokens=draft_tokens,
                    target_tokens=target_tokens[:len(draft_tokens)],
                    accepted_mask=accepted_mask,
                    anchor_pos=anchor_pos,
                    source="online",
                )

            # Update stats
            stats["n_calls"] += 1
            stats["n_accepted"] += n_accepted
            stats["n_drafted"] += len(draft_tokens)
            self.total_accepted += n_accepted
            self.total_drafted += len(draft_tokens)
            self.total_calls += 1

            # Check for EOS
            if generated and generated[-1] in (151643, 151645):  # Qwen EOS tokens
                break

        # Detokenize
        output_text = self.target.detokenize(generated)

        stats["tau"] = stats["n_accepted"] / max(1, stats["n_drafted"])
        stats["tokens_generated"] = len(generated)
        stats["avg_draft_ms"] = stats["draft_ms"] / max(1, stats["n_calls"])

        return output_text, stats

    @property
    def running_tau(self) -> float:
        return self.total_accepted / max(1, self.total_drafted)
