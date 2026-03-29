"""Python wrapper for the C++ target daemon subprocess.

Binary protocol: [cmd:u8][payload_len:u32_le][payload...] → [status:u8][payload_len:u32_le][payload...]
"""

import os
import select
import struct
import subprocess
import sys
from typing import List, Optional, Tuple

import numpy as np

CMD_TOKENIZE = 0x01
CMD_EVAL_FULL = 0x02
CMD_EVAL_INCR = 0x03
CMD_TRIM_KV = 0x04
CMD_CLEAR_KV = 0x05
CMD_DETOKENIZE = 0x06
CMD_EVAL_LAST = 0x07
CMD_STATE_SAVE = 0x08
CMD_STATE_RESTORE = 0x09
CMD_GET_ROUTING = 0x0A
CMD_STATE_SAVE_GDN = 0x0B
CMD_STATE_RESTORE_GDN = 0x0C
CMD_QUIT = 0xFF

STATUS_OK = 0x00
STATUS_ERROR = 0x01

HEADER_FMT = "<BI"  # cmd:u8 + payload_len:u32_le
HEADER_SIZE = struct.calcsize(HEADER_FMT)


class TargetDaemon:
    """Manages C++ target model subprocess for online speculative decoding."""

    def __init__(
        self,
        daemon_path: str,
        model_path: str,
        layers: List[int],
        n_gpu_layers: int = 99,
        extra_args: Optional[List[str]] = None,
        capture_routing: bool = False,
        routing_layers: Optional[List[int]] = None,
    ):
        """Launch the C++ daemon subprocess.

        Args:
            daemon_path: path to target_daemon binary
            model_path: path to GGUF model
            layers: list of int layer IDs [2, 11, 20, 29, 37]
            n_gpu_layers: GPU offload layers
            extra_args: list of extra CLI args for llama.cpp
            capture_routing: if True, pass --capture-routing to daemon
            routing_layers: explicit list of MoE layers to capture routing for.
                If None and capture_routing=True, daemon defaults to --layers.
        """
        self.layers = layers
        self.capture_routing = capture_routing
        self.routing_layers = routing_layers if routing_layers is not None else layers
        cmd = [
            daemon_path,
            "-m", model_path,
            "-ngl", str(n_gpu_layers),
            "--layers", ",".join(str(l) for l in layers),
        ]
        if capture_routing:
            cmd.append("--capture-routing")
            if routing_layers is not None:
                cmd.extend(["--routing-layers", ",".join(str(l) for l in routing_layers)])
        if extra_args:
            cmd.extend(extra_args)
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            bufsize=0,
        )

    # ── low-level I/O ──────────────────────────────────────────────

    def _read_exact(self, n: int, timeout: float = 120.0) -> bytes:
        buf = bytearray()
        fd = self._proc.stdout.fileno()
        while len(buf) < n:
            ready, _, _ = select.select([fd], [], [], timeout)
            if not ready:
                raise TimeoutError(f"daemon read timeout ({timeout}s)")
            chunk = os.read(fd, n - len(buf))
            if not chunk:
                raise ConnectionError("daemon stdout closed unexpectedly")
            buf.extend(chunk)
        return bytes(buf)

    def _send_cmd(self, cmd: int, payload: bytes = b"") -> None:
        header = struct.pack(HEADER_FMT, cmd, len(payload))
        self._proc.stdin.write(header + payload)
        self._proc.stdin.flush()

    def _recv_response(self) -> Tuple[int, bytes]:
        header = self._read_exact(HEADER_SIZE)
        status, length = struct.unpack(HEADER_FMT, header)
        payload = self._read_exact(length) if length > 0 else b""
        if status == STATUS_ERROR:
            raise RuntimeError(f"daemon error: {payload.decode('utf-8', errors='replace')}")
        return status, payload

    # ── public API ─────────────────────────────────────────────────

    def tokenize(self, text: str) -> List[int]:
        """Tokenize UTF-8 text into token IDs."""
        self._send_cmd(CMD_TOKENIZE, text.encode("utf-8"))
        _, payload = self._recv_response()
        return list(struct.unpack(f"<{len(payload) // 4}i", payload))

    def detokenize(self, tokens: List[int]) -> str:
        """Detokenize token IDs into UTF-8 text."""
        payload = struct.pack(f"<{len(tokens)}i", *tokens)
        self._send_cmd(CMD_DETOKENIZE, payload)
        _, resp = self._recv_response()
        return resp.decode("utf-8")

    def eval_full(
        self, tokens: List[int], layer_ids: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Full eval with KV cache clear.

        Returns:
            hidden_states: float32 array [n_layers, seq_len, hidden_dim]
            argmax_tokens: int32 array [seq_len]
        """
        return self._eval(CMD_EVAL_FULL, tokens, layer_ids)

    def eval_last_pos(
        self, tokens: List[int], layer_ids: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, int]:
        """Full eval returning ONLY the last position's hidden states.

        Much faster than eval_full for extraction (40 KB vs 20 MB pipe I/O).

        Returns:
            hidden_states: float32 array [n_layers, 1, hidden_dim]
            argmax_token: int (argmax of last position)
        """
        if layer_ids is None:
            layer_ids = self.layers
        payload = struct.pack(f"<I{len(tokens)}i", len(tokens), *tokens)
        payload += struct.pack(f"<I{len(layer_ids)}i", len(layer_ids), *layer_ids)
        self._send_cmd(CMD_EVAL_LAST, payload)
        _, resp = self._recv_response()

        off = 0
        n_layers, seq_len, hidden_dim = struct.unpack_from("<III", resp, off)
        off += 12
        n_floats = n_layers * seq_len * hidden_dim
        hidden = np.frombuffer(resp, dtype=np.float32, count=n_floats, offset=off)
        hidden = hidden.reshape(n_layers, seq_len, hidden_dim)
        off += n_floats * 4
        argmax = struct.unpack_from("<i", resp, off)[0]
        return hidden.copy(), argmax

    def eval_incr(
        self, tokens: List[int], layer_ids: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Incremental eval (appends to KV cache).

        Returns:
            hidden_states: float32 array [n_layers, n_tokens, hidden_dim]
            argmax_tokens: int32 array [n_tokens]
        """
        return self._eval(CMD_EVAL_INCR, tokens, layer_ids)

    def _eval(
        self, cmd: int, tokens: List[int], layer_ids: Optional[List[int]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        if layer_ids is None:
            layer_ids = self.layers
        payload = struct.pack(f"<I{len(tokens)}i", len(tokens), *tokens)
        payload += struct.pack(f"<I{len(layer_ids)}i", len(layer_ids), *layer_ids)
        self._send_cmd(cmd, payload)
        _, resp = self._recv_response()

        off = 0
        n_layers, seq_len, hidden_dim = struct.unpack_from("<III", resp, off)
        off += 12
        n_floats = n_layers * seq_len * hidden_dim
        hidden = np.frombuffer(resp, dtype=np.float32, count=n_floats, offset=off)
        hidden = hidden.reshape(n_layers, seq_len, hidden_dim)
        off += n_floats * 4
        logits = np.frombuffer(resp, dtype=np.int32, count=seq_len, offset=off)
        return hidden.copy(), logits.copy()

    def get_routing(self) -> np.ndarray:
        """Get MoE routing decisions from the most recent eval.

        Requires the daemon to have been started with capture_routing=True.

        Returns:
            routing: uint8 array of shape [n_routing_layers, n_tokens, n_expert_used]
                where n_expert_used = 8 for Qwen3.5-35B-A3B.
                routing[i, t, :] gives the 8 expert indices selected for token t
                at the i-th routing layer (layers in ascending order).

        Raises:
            RuntimeError: if routing capture was not enabled or no eval has been run.
        """
        self._send_cmd(CMD_GET_ROUTING)
        _, payload = self._recv_response()

        if len(payload) < 4:
            raise RuntimeError(f"GET_ROUTING response too short: {len(payload)} bytes")

        # Parse header: n_layers(u16 LE) + n_tokens(u16 LE)
        n_layers, n_tokens = struct.unpack_from("<HH", payload, 0)
        offset = 4

        # Determine n_expert_used from remaining data
        data_bytes = len(payload) - offset
        expected_min = n_layers * n_tokens  # at least 1 expert per (layer, token)
        if data_bytes < expected_min:
            raise RuntimeError(
                f"GET_ROUTING payload too short: got {data_bytes} bytes, "
                f"expected at least {expected_min} for {n_layers} layers × {n_tokens} tokens"
            )

        # n_expert_used = data_bytes / (n_layers * n_tokens)
        n_expert_used = data_bytes // (n_layers * n_tokens) if (n_layers * n_tokens) > 0 else 0
        if n_layers * n_tokens * n_expert_used != data_bytes:
            raise RuntimeError(
                f"GET_ROUTING data size mismatch: {data_bytes} bytes not divisible by "
                f"{n_layers} × {n_tokens} = {n_layers * n_tokens}"
            )

        routing = np.frombuffer(payload, dtype=np.uint8, count=data_bytes, offset=offset)
        routing = routing.reshape(n_layers, n_tokens, n_expert_used).copy()
        return routing

    def trim_kv(self, keep_n: int) -> None:
        """Trim KV cache to keep_n tokens."""
        self._send_cmd(CMD_TRIM_KV, struct.pack("<i", keep_n))
        self._recv_response()

    def clear_kv(self) -> None:
        """Clear KV cache entirely."""
        self._send_cmd(CMD_CLEAR_KV)
        self._recv_response()

    def save_state(self) -> int:
        """Save full recurrent+KV state (single slot). Returns state size in bytes."""
        self._send_cmd(CMD_STATE_SAVE)
        _, payload = self._recv_response()
        return struct.unpack("<I", payload)[0]

    def restore_state(self) -> None:
        """Restore previously saved recurrent+KV state."""
        self._send_cmd(CMD_STATE_RESTORE)
        self._recv_response()

    def save_state_gdn(self) -> int:
        """Save ONLY GDN recurrent states (~2 MB vs ~63 MB full state).

        Skips attention KV cache (handled separately via trim_kv).
        Returns state size in bytes.
        """
        self._send_cmd(CMD_STATE_SAVE_GDN)
        _, payload = self._recv_response()
        return struct.unpack("<I", payload)[0]

    def restore_state_gdn(self, keep_n: int) -> None:
        """Restore GDN recurrent states + trim attention KV cache.

        Args:
            keep_n: number of KV positions to keep in attention cache.
                    Positions [keep_n, ...) are removed.
        """
        self._send_cmd(CMD_STATE_RESTORE_GDN, struct.pack("<i", keep_n))
        self._recv_response()

    def close(self) -> None:
        """Send QUIT and wait for process to exit."""
        if self._proc and self._proc.poll() is None:
            try:
                self._send_cmd(CMD_QUIT)
                self._proc.wait(timeout=10)
            except (BrokenPipeError, OSError):
                self._proc.kill()
                self._proc.wait()
        self._proc = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
