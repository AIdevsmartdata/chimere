"""
Speculative decoding: draft with DFlash, verify with target AR model.

Two operating modes:
  1. Offline benchmark — uses pre-extracted hidden states to measure acceptance rate
  2. Online (future) — requires custom llama-server endpoint for live hidden states

Verification algorithm from Leviathan et al. 2023 / Chen et al. 2023:
  - Draft generates K tokens in parallel (single diffusion forward pass)
  - Target verifies K tokens in single forward pass
  - Accept sequentially left-to-right until first mismatch
  - On rejection: resample from residual distribution
  - All K accepted → bonus token from target
"""

import json
import time
from dataclasses import dataclass

import numpy as np
import requests
import torch
import torch.nn.functional as F

from .config import DFlashConfig
from .modeling import DFlashDraftModel
from .modeling_v5 import DFlashDraftModelV5


@dataclass
class SpecDecodeStats:
    """Statistics from one generation run."""
    total_tokens: int = 0
    total_steps: int = 0
    total_drafted: int = 0
    total_accepted: int = 0
    total_target_calls: int = 0
    draft_time_ms: float = 0.0
    verify_time_ms: float = 0.0
    total_time_ms: float = 0.0

    @property
    def acceptance_rate(self):
        return self.total_accepted / max(1, self.total_drafted)

    @property
    def tokens_per_step(self):
        return self.total_tokens / max(1, self.total_steps)

    @property
    def speedup_vs_ar(self):
        """Estimated speedup over pure autoregressive decoding."""
        # AR would need total_tokens target calls
        # We used total_target_calls target calls
        return self.total_tokens / max(1, self.total_target_calls)

    def __repr__(self):
        return (
            f"SpecDecodeStats(tokens={self.total_tokens}, steps={self.total_steps}, "
            f"accept_rate={self.acceptance_rate:.2%}, "
            f"tokens/step={self.tokens_per_step:.1f}, "
            f"speedup={self.speedup_vs_ar:.2f}x, "
            f"draft={self.draft_time_ms:.0f}ms, verify={self.verify_time_ms:.0f}ms)"
        )


class TargetModelAPI:
    """Interface to target model running on llama-server."""

    def __init__(self, base_url="http://localhost:8081", timeout=120):
        self.base_url = base_url
        self.timeout = timeout

    def verify_tokens(self, prompt_tokens, draft_tokens, temperature=0.0):
        """
        Verify draft tokens by running target model on prompt + draft.

        Returns target's top-1 predictions and logprobs at each draft position.
        Uses /completion endpoint with the full sequence to get logprobs.
        """
        # Tokenize prompt + draft tokens, then evaluate with logprobs
        # llama-server can evaluate a prompt and return logprobs for each token
        all_tokens = prompt_tokens + draft_tokens

        resp = requests.post(
            f"{self.base_url}/completion",
            json={
                "prompt": all_tokens,
                "n_predict": 1,  # generate 1 bonus token
                "temperature": temperature,
                "logprobs": True,
                "n_probs": 10,  # top-10 for approximate verification
                "cache_prompt": True,
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        return data

    def tokenize(self, text):
        """Tokenize text via llama-server API."""
        resp = requests.post(
            f"{self.base_url}/tokenize",
            json={"content": text},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["tokens"]

    def detokenize(self, tokens):
        """Detokenize tokens via llama-server API."""
        resp = requests.post(
            f"{self.base_url}/detokenize",
            json={"tokens": tokens},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["content"]


class SpeculativeDecoder:
    """
    Main speculative decoding loop.

    Inference cycle:
    1. Get target hidden states at current position
    2. DFlash drafts block of K tokens in parallel (single forward pass)
    3. Target verifies draft tokens
    4. Accept tokens until first mismatch, resample on rejection
    5. Repeat until max_tokens or EOS
    """

    def __init__(
        self,
        drafter: DFlashDraftModel,
        config: DFlashConfig = None,
        target_url: str = "http://localhost:8081",
        device: str = "cuda",
    ):
        self.drafter = drafter
        self.config = config or drafter.config
        self.target = TargetModelAPI(target_url)
        self.device = torch.device(device)
        self.block_size = self.config.block_size

    @torch.no_grad()
    def draft_block(self, hidden_states_list, temperature=0.0, n_steps=1,
                    anchor_token_id=None):
        """
        Generate a block of K draft tokens using the DFlash drafter.

        Args:
            hidden_states_list: list of k tensors [1, seq_len, hidden_dim]
                from target layers at the current prefix
            temperature: sampling temperature (0 = greedy)
            n_steps: number of denoising steps (1 = single-shot DFlash default)
            anchor_token_id: int or None — verified token for position 0

        Returns:
            draft_ids: [K] int64 — drafted token IDs
            draft_logits: [K, vocab] float — draft model's logit distribution
        """
        draft_ids, draft_logits = self.drafter.generate_block(
            hidden_states_list, temperature=temperature, n_steps=n_steps,
            anchor_token_id=anchor_token_id,
        )
        return draft_ids[0], draft_logits[0]  # remove batch dim

    def verify_greedy(self, prompt_tokens, draft_tokens):
        """
        Greedy verification (temperature=0): accept if target's argmax matches draft.

        This is simpler and doesn't require full probability distributions.
        Returns (accepted_tokens, bonus_token, n_accepted).
        """
        n_draft = len(draft_tokens)
        all_tokens = prompt_tokens + draft_tokens

        # Single target forward pass: evaluate prompt+draft, generate 1 more
        resp = requests.post(
            f"{self.target.base_url}/completion",
            json={
                "prompt": all_tokens,
                "n_predict": 1,
                "temperature": 0.0,
                "cache_prompt": True,
            },
            timeout=self.target.timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        # The target model processed all_tokens and generated 1 more
        # For greedy verification, we need the target's predictions at each
        # draft position. With llama-server, we can use prompt logprobs.
        # However, standard llama-server doesn't return per-position logprobs
        # for the prompt evaluation.
        #
        # Workaround: evaluate incrementally position by position.
        # This is slower but correct for the MVP.
        accepted = []
        for i in range(n_draft):
            prefix = prompt_tokens + draft_tokens[:i]
            resp = requests.post(
                f"{self.target.base_url}/completion",
                json={
                    "prompt": prefix,
                    "n_predict": 1,
                    "temperature": 0.0,
                    "cache_prompt": True,
                },
                timeout=self.target.timeout,
            )
            resp.raise_for_status()
            result = resp.json()

            # Get the token the target would have generated
            target_tokens = result.get("tokens", [])
            if not target_tokens:
                # Parse from content
                target_text = result.get("content", "")
                target_tokens = self.target.tokenize(target_text) if target_text else []

            if target_tokens and target_tokens[0] == draft_tokens[i]:
                accepted.append(draft_tokens[i])
            else:
                # Reject: use target's token instead
                if target_tokens:
                    accepted.append(target_tokens[0])
                break

        # Bonus token if all accepted
        bonus = None
        if len(accepted) == n_draft:
            prefix = prompt_tokens + draft_tokens
            resp = requests.post(
                f"{self.target.base_url}/completion",
                json={
                    "prompt": prefix,
                    "n_predict": 1,
                    "temperature": 0.0,
                    "cache_prompt": True,
                },
                timeout=self.target.timeout,
            )
            resp.raise_for_status()
            result = resp.json()
            target_text = result.get("content", "")
            bonus_tokens = self.target.tokenize(target_text) if target_text else []
            if bonus_tokens:
                bonus = bonus_tokens[0]

        n_accepted = min(len(accepted), n_draft)
        if len(accepted) == n_draft and accepted == draft_tokens:
            n_accepted = n_draft  # all accepted
        else:
            # Count actual matches (accepted includes the replacement token)
            n_accepted = 0
            for i in range(min(len(accepted), n_draft)):
                if accepted[i] == draft_tokens[i]:
                    n_accepted += 1
                else:
                    break

        return accepted, bonus, n_accepted

    def benchmark_offline(
        self,
        prompt_tokens,
        block_hidden_states,
        ground_truth_tokens,
        temperature=0.0,
        n_steps=1,
    ):
        """
        Offline benchmark using pre-extracted hidden states.

        Simulates the speculative decoding loop without running the target model.
        Uses ground truth tokens (from the actual target generation) to measure
        what the acceptance rate WOULD be.

        Args:
            prompt_tokens: list of int — prompt token IDs
            block_hidden_states: list of dicts, each containing
                'block_hidden' [k, block_size, hidden_dim] and
                'block_input_ids' [block_size]
            ground_truth_tokens: list of int — what the target actually generated
            temperature: float
            n_steps: number of denoising steps for drafting

        Returns:
            SpecDecodeStats
        """
        stats = SpecDecodeStats()
        pos = 0
        total_gt = len(ground_truth_tokens)
        k = self.config.num_feature_layers

        t_start = time.time()

        for block_data in block_hidden_states:
            if pos >= total_gt:
                break

            block_hidden = block_data["block_hidden"]  # [k, block_size, hidden_dim]
            gt_ids = block_data["block_input_ids"]      # [block_size]
            remaining = total_gt - pos
            actual_block = min(self.block_size, remaining)

            # Unpack to list of k tensors [1, block_size, hidden_dim]
            hidden_list = [
                block_hidden[i:i+1].unsqueeze(0).to(self.device)
                if block_hidden.dim() == 3
                else block_hidden[i].unsqueeze(0).unsqueeze(0).to(self.device)
                for i in range(k)
            ]

            # Fix shapes: we need [1, block_size, hidden_dim] per layer
            hidden_list_fixed = []
            for i in range(k):
                h = block_hidden[i]  # [block_size, hidden_dim]
                hidden_list_fixed.append(h.unsqueeze(0).to(self.device))  # [1, S, H]

            # Get anchor token (position 0 = verified token from previous block)
            anchor_token_id = int(gt_ids[0].item())

            # Draft
            t_draft = time.time()
            draft_ids, draft_logits = self.draft_block(
                hidden_list_fixed, temperature=temperature, n_steps=n_steps,
                anchor_token_id=anchor_token_id,
            )
            stats.draft_time_ms += (time.time() - t_draft) * 1000

            # Verify against ground truth (offline — no target call needed)
            # Position 0 is the anchor (already verified), so only check positions 1..K-1
            t_verify = time.time()
            draft_ids_cpu = draft_ids[:actual_block].cpu()
            gt_block = gt_ids[:actual_block]

            n_accepted = 0
            # Skip position 0 (anchor) — it's always correct by construction
            for j in range(1, actual_block):
                if draft_ids_cpu[j].item() == gt_block[j].item():
                    n_accepted += 1
                else:
                    break

            stats.verify_time_ms += (time.time() - t_verify) * 1000

            # In real spec decode: accepted tokens + 1 target call for rejection/bonus
            tokens_produced = n_accepted + 1  # +1 for resample or bonus
            tokens_produced = min(tokens_produced, remaining)

            stats.total_drafted += actual_block - 1  # exclude anchor at position 0
            stats.total_accepted += n_accepted
            stats.total_tokens += tokens_produced
            stats.total_steps += 1
            stats.total_target_calls += 1  # 1 target call per step

            pos += tokens_produced

        stats.total_time_ms = (time.time() - t_start) * 1000

        return stats

    @torch.no_grad()
    def generate_online(self, prompt_text, max_new_tokens=256, temperature=0.0,
                        stop_token_ids=None, target_daemon=None):
        """
        Online speculative decoding with C++ target daemon.

        Args:
            prompt_text: input text
            max_new_tokens: max tokens to generate
            temperature: drafting temperature
            stop_token_ids: set of stop token IDs (default: {248046} = <|im_end|>)
            target_daemon: TargetDaemon instance (required for online mode)

        Returns:
            (generated_text, SpecDecodeStats)
        """
        if target_daemon is None:
            raise ValueError("target_daemon required for online mode")
        if stop_token_ids is None:
            stop_token_ids = {248046}

        stats = SpecDecodeStats()
        t_start = time.time()

        prompt_tokens = target_daemon.tokenize(prompt_text)
        n_prompt = len(prompt_tokens)

        # Full eval of prompt — get hidden states and target predictions
        hidden, logits = target_daemon.eval_full(prompt_tokens)
        # hidden: np.ndarray [n_layers, seq_len, hidden_dim]
        # logits: np.ndarray [seq_len] (argmax token IDs)

        generated = []
        k = self.config.num_feature_layers
        block_size = self.block_size
        kv_pos = n_prompt  # tokens currently in KV cache

        # Track last hidden states for drafting context
        last_hidden = hidden  # full hidden from last eval

        while len(generated) < max_new_tokens:
            stats.total_steps += 1

            # Build hidden context for drafter: last block_size positions from hidden
            n_avail = last_hidden.shape[1]
            n_ctx = min(block_size, n_avail)
            hs = last_hidden[:, -n_ctx:, :]  # [k, n_ctx, H]

            if n_ctx < block_size:
                pad = np.zeros(
                    (k, block_size - n_ctx, last_hidden.shape[2]),
                    dtype=np.float32,
                )
                hs = np.concatenate([pad, hs], axis=1)

            hidden_list = [
                torch.from_numpy(hs[i].copy())
                .unsqueeze(0)
                .to(self.device, dtype=torch.float32)
                for i in range(k)
            ]

            # Anchor = target's prediction at last position
            anchor_id = int(logits[-1])

            # Check EOS
            if anchor_id in stop_token_ids:
                break

            # Draft K tokens
            t_draft = time.time()
            draft_ids, _ = self.draft_block(
                hidden_list, temperature=temperature, n_steps=1,
                anchor_token_id=anchor_id,
            )
            draft_tokens = draft_ids.cpu().tolist()  # [block_size], pos 0 = anchor
            stats.draft_time_ms += (time.time() - t_draft) * 1000
            stats.total_drafted += block_size - 1  # exclude anchor

            # Verify: eval_incr ALL draft tokens through target
            # draft_tokens[0] = anchor (not yet in KV), draft_tokens[1..K-1] = speculative
            t_verify = time.time()
            v_hidden, v_logits = target_daemon.eval_incr(draft_tokens)
            # v_logits[i] = target's argmax after absorbing draft_tokens[0..i]
            # v_hidden: [n_layers, K, hidden_dim]
            stats.verify_time_ms += (time.time() - t_verify) * 1000
            stats.total_target_calls += 1

            # Verify left-to-right: v_logits[j] should == draft_tokens[j+1]
            n_accepted = 0
            for j in range(block_size - 1):
                if int(v_logits[j]) == draft_tokens[j + 1]:
                    n_accepted += 1
                else:
                    break

            stats.total_accepted += n_accepted

            if n_accepted == block_size - 1:
                # All draft tokens accepted! Add all + bonus
                new_tokens = draft_tokens + [int(v_logits[-1])]
                generated.extend(new_tokens)
                kv_pos += block_size
                # Update hidden/logits from verification eval
                last_hidden = v_hidden
                logits = v_logits
            else:
                # Partial acceptance: keep draft[0..n_accepted] + target's correction
                accepted_tokens = (
                    draft_tokens[:n_accepted + 1]
                    + [int(v_logits[n_accepted])]
                )
                generated.extend(accepted_tokens)
                # KV has rejected draft tokens — M-RoPE forbids position backtracking.
                # Clear and re-eval the entire accepted prefix (excluding the
                # correction token which becomes the next anchor, not yet in KV).
                prefix = prompt_tokens + generated[:-1]
                target_daemon.clear_kv()
                last_hidden, logits = target_daemon.eval_full(prefix)
                kv_pos = len(prefix)

            stats.total_tokens = len(generated)

            # Check for EOS in generated tokens
            if any(t in stop_token_ids for t in generated[-block_size:]):
                # Trim to EOS
                for eos_pos, t in enumerate(generated):
                    if t in stop_token_ids:
                        generated = generated[:eos_pos]
                        break
                break

        # Truncate to max
        generated = generated[:max_new_tokens]
        stats.total_tokens = len(generated)
        stats.total_time_ms = (time.time() - t_start) * 1000

        text = target_daemon.detokenize(generated) if generated else ""
        return text, stats

    @torch.no_grad()
    def generate_ar_daemon(self, prompt_text, max_new_tokens=256, temperature=0.0,
                           stop_token_ids=None, target_daemon=None):
        """
        Pure autoregressive baseline using C++ target daemon.
        Fair comparison: same daemon, same model, no drafting.
        """
        if target_daemon is None:
            raise ValueError("target_daemon required")
        if stop_token_ids is None:
            stop_token_ids = {248046}

        stats = SpecDecodeStats()
        t_start = time.time()

        prompt_tokens = target_daemon.tokenize(prompt_text)
        _, logits = target_daemon.eval_full(prompt_tokens)

        generated = []

        while len(generated) < max_new_tokens:
            next_token = int(logits[-1])
            if next_token in stop_token_ids:
                break
            generated.append(next_token)
            stats.total_target_calls += 1

            _, logits = target_daemon.eval_incr([next_token])

        stats.total_tokens = len(generated)
        stats.total_steps = len(generated)
        stats.total_time_ms = (time.time() - t_start) * 1000

        text = target_daemon.detokenize(generated) if generated else ""
        return text, stats

    def generate_ar_baseline(
        self,
        prompt_text,
        max_new_tokens=256,
        temperature=0.0,
    ):
        """
        Pure autoregressive baseline for comparison.
        Generates tokens one by one via target model API.
        """
        stats = SpecDecodeStats()
        t_start = time.time()

        resp = requests.post(
            f"{self.target.base_url}/v1/chat/completions",
            json={
                "model": "qwen3.5",
                "messages": [{"role": "user", "content": prompt_text}],
                "max_tokens": max_new_tokens,
                "temperature": temperature,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=self.target.timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        content = data["choices"][0]["message"].get("content", "")
        usage = data.get("usage", {})

        stats.total_tokens = usage.get("completion_tokens", 0)
        stats.total_steps = stats.total_tokens  # 1 token per step in AR
        stats.total_target_calls = stats.total_tokens
        stats.total_time_ms = (time.time() - t_start) * 1000

        return content, stats


@torch.no_grad()
def generate_online_v5(drafter_v5, prompt_text, max_new_tokens=256, temperature=0.0,
                       stop_token_ids=None, target_daemon=None, max_ctx_len=512,
                       n_denoise_steps=1):
    """
    Online speculative decoding with DFlash v5 (full context KV injection).

    Unlike v3/v4 which only used the last block_size hidden states,
    v5 accumulates ALL hidden states from the verified prefix and
    cross-attends to them at every drafter layer.

    Args:
        drafter_v5: DFlashDraftModelV5 instance (on GPU)
        prompt_text: input text
        max_new_tokens: max tokens to generate
        temperature: drafting temperature
        stop_token_ids: set of stop token IDs
        target_daemon: TargetDaemon instance
        max_ctx_len: max context length for drafter attention

    Returns:
        (generated_text, SpecDecodeStats)
    """
    if target_daemon is None:
        raise ValueError("target_daemon required for online mode")
    if stop_token_ids is None:
        stop_token_ids = {248046}

    config = drafter_v5.config
    block_size = config.block_size
    k = config.num_feature_layers
    device = next(drafter_v5.parameters()).device

    stats = SpecDecodeStats()
    t_start = time.time()

    prompt_tokens = target_daemon.tokenize(prompt_text)
    n_prompt = len(prompt_tokens)

    # Full eval of prompt
    hidden, logits = target_daemon.eval_full(prompt_tokens)
    # hidden: [n_layers, seq_len, hidden_dim]

    generated = []
    kv_pos = n_prompt

    # Accumulate ALL hidden states from verified positions
    # hidden is [n_layers, n_prompt, hidden_dim]
    all_hidden = hidden  # numpy array

    while len(generated) < max_new_tokens:
        stats.total_steps += 1

        # Build context: all accumulated hidden states (clipped to max_ctx_len)
        n_ctx = all_hidden.shape[1]
        ctx_start = max(0, n_ctx - max_ctx_len)
        ctx = all_hidden[:, ctx_start:, :]  # [n_layers, ctx_len, hidden_dim]
        actual_ctx_len = ctx.shape[1]

        # Convert to list of tensors for drafter
        hidden_list = [
            torch.from_numpy(ctx[i].copy())
            .unsqueeze(0)
            .to(device, dtype=torch.float32)
            for i in range(k)
        ]
        ctx_lengths = torch.tensor([actual_ctx_len], device=device, dtype=torch.long)

        # Anchor = target's prediction at last verified position
        anchor_id = int(logits[-1])

        if anchor_id in stop_token_ids:
            break

        # Draft K tokens
        t_draft = time.time()
        if n_denoise_steps > 1:
            draft_ids, _ = drafter_v5.generate_block_multistep(
                hidden_list, context_lengths=ctx_lengths,
                temperature=temperature, anchor_token_id=anchor_id,
                n_steps=n_denoise_steps,
            )
        else:
            draft_ids, _ = drafter_v5.generate_block(
                hidden_list, context_lengths=ctx_lengths,
                temperature=temperature, anchor_token_id=anchor_id,
            )
        draft_tokens = draft_ids[0].cpu().tolist()  # [block_size]
        stats.draft_time_ms += (time.time() - t_draft) * 1000
        stats.total_drafted += block_size - 1  # exclude anchor

        # Verify: eval_incr ALL draft tokens through target
        t_verify = time.time()
        v_hidden, v_logits = target_daemon.eval_incr(draft_tokens)
        stats.verify_time_ms += (time.time() - t_verify) * 1000
        stats.total_target_calls += 1

        # Verify left-to-right
        n_accepted = 0
        for j in range(block_size - 1):
            if int(v_logits[j]) == draft_tokens[j + 1]:
                n_accepted += 1
            else:
                break

        stats.total_accepted += n_accepted

        if n_accepted == block_size - 1:
            # All accepted + bonus
            new_tokens = draft_tokens + [int(v_logits[-1])]
            generated.extend(new_tokens)
            kv_pos += block_size
            # Accumulate ALL verified hidden states
            all_hidden = np.concatenate([all_hidden, v_hidden], axis=1)
            logits = v_logits
        else:
            # Partial acceptance
            accepted_tokens = (
                draft_tokens[:n_accepted + 1]
                + [int(v_logits[n_accepted])]
            )
            generated.extend(accepted_tokens)
            # M-RoPE requires monotonic positions — cannot trim_kv and go back.
            # Must clear KV and re-eval entire prefix to get correct hidden states.
            prefix = prompt_tokens + generated
            target_daemon.clear_kv()
            all_hidden, logits = target_daemon.eval_full(prefix)
            kv_pos = len(prefix)

        stats.total_tokens = len(generated)

        # Check EOS
        if any(t in stop_token_ids for t in generated[-block_size:]):
            for eos_pos, t in enumerate(generated):
                if t in stop_token_ids:
                    generated = generated[:eos_pos]
                    break
            break

    generated = generated[:max_new_tokens]
    stats.total_tokens = len(generated)
    stats.total_time_ms = (time.time() - t_start) * 1000

    text = target_daemon.detokenize(generated) if generated else ""
    return text, stats
