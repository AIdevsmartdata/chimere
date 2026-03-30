"""
title: Chimere Quality Gate
author: chimere-stack
version: 1.0.0
description: Filter function that scores each LLM response via the Chimere quality gate (ThinkPRM or Qwen3.5 heuristic scorer). Logs scores to quality_scores.jsonl and adds a low-confidence indicator when score <= 2.
license: Apache-2.0
"""

import json
import time
import threading
import urllib.request
import urllib.error
from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, Field


class Filter:
    """Open WebUI Filter that runs post-response quality scoring.

    After each LLM response, this filter:
    1. Sends the prompt+response pair to the quality scorer (port 8085 ThinkPRM
       or port 8081 Qwen3.5 heuristic)
    2. Logs the score to quality_scores.jsonl
    3. If score <= 2, appends a low-confidence warning to the response
    4. If score >= 4, marks the response as an Engram write candidate

    The scoring runs asynchronously in a background thread to avoid
    blocking the response delivery.
    """

    class Valves(BaseModel):
        QUALITY_SCORER_URL: str = Field(
            default="http://odo:8084/v1/chat/completions",
            description="URL of the quality scorer (ODO proxies to ThinkPRM/Qwen3.5)",
        )
        THINKPRM_URL: str = Field(
            default="http://odo:8085/v1/chat/completions",
            description="Direct URL of the ThinkPRM scorer (if available)",
        )
        LOG_PATH: str = Field(
            default="/data/logs/quality_scores.jsonl",
            description="Path to the quality scores log file (inside Open WebUI container)",
        )
        LOW_SCORE_THRESHOLD: int = Field(
            default=2,
            description="Score at or below which the low-confidence indicator is shown",
        )
        HIGH_SCORE_THRESHOLD: int = Field(
            default=4,
            description="Score at or above which the response is flagged as Engram candidate",
        )
        MIN_RESPONSE_LENGTH: int = Field(
            default=100,
            description="Minimum response length (chars) to trigger scoring",
        )
        SCORE_TIMEOUT: int = Field(
            default=30,
            description="Timeout in seconds for the scoring request",
        )
        ENABLED: bool = Field(
            default=True,
            description="Enable/disable quality scoring",
        )
        SHOW_SCORE_BADGE: bool = Field(
            default=False,
            description="Show the quality score as a small badge on every response (debug mode)",
        )

    def __init__(self):
        self.valves = self.Valves()

    def inlet(self, body: dict) -> dict:
        """Pass through on inlet -- no input modification needed."""
        return body

    def outlet(self, body: dict) -> None:
        """Post-process the LLM response: score quality in background.

        The outlet function receives the full conversation including the
        latest assistant response. We extract it and score asynchronously.
        """

        if not self.valves.ENABLED:
            return

        messages = body.get("messages", [])
        if not messages:
            return

        # Find the last assistant message
        assistant_msg = None
        user_msg = None
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and assistant_msg is None:
                assistant_msg = msg.get("content", "")
            elif msg.get("role") == "user" and user_msg is None:
                user_msg = msg.get("content", "")
            if assistant_msg is not None and user_msg is not None:
                break

        if not assistant_msg or len(assistant_msg) < self.valves.MIN_RESPONSE_LENGTH:
            return

        if not user_msg:
            return

        # Score in background to not block the response
        thread = threading.Thread(
            target=self._score_and_log,
            args=(user_msg, assistant_msg, body),
            daemon=True,
        )
        thread.start()

    def stream(self, event: dict) -> dict:
        """Pass through stream events -- we process in outlet after completion."""
        return event

    def _score_and_log(self, user_msg: str, assistant_msg: str, body: dict) -> None:
        """Background thread: call the quality scorer and log results."""

        score = self._get_quality_score(user_msg, assistant_msg)
        if score is None:
            return

        # Log to file
        self._log_score(user_msg, assistant_msg, score, body)

        # Modify the last assistant message in-place if needed
        messages = body.get("messages", [])
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")

                if score <= self.valves.LOW_SCORE_THRESHOLD:
                    msg["content"] = (
                        content + "\n\n---\n*Low confidence response (quality score: "
                        f"{score}/5). Consider verifying this information.*"
                    )
                elif self.valves.SHOW_SCORE_BADGE:
                    badge = self._score_to_badge(score)
                    msg["content"] = content + f"\n\n{badge}"

                break

    def _get_quality_score(self, user_msg: str, assistant_msg: str) -> Optional[int]:
        """Call the quality scorer and return a 1-5 score."""

        # Try ThinkPRM first (port 8085)
        score = self._try_thinkprm(user_msg, assistant_msg)
        if score is not None:
            return score

        # Fallback to Qwen3.5 heuristic scorer
        return self._try_qwen_scorer(user_msg, assistant_msg)

    def _try_thinkprm(self, user_msg: str, assistant_msg: str) -> Optional[int]:
        """Try scoring via ThinkPRM-1.5B."""
        try:
            payload = {
                "model": "thinkprm",
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"Rate this response on a scale of 1-5.\n\n"
                            f"Question: {user_msg[:1000]}\n\n"
                            f"Response: {assistant_msg[:2000]}\n\n"
                            f"Reply with ONLY a JSON: {{\"score\": N, \"reason\": \"...\"}}"
                        ),
                    }
                ],
                "max_tokens": 256,
                "temperature": 0.1,
                "stream": False,
            }

            url = self.valves.THINKPRM_URL
            data = json.dumps(payload, ensure_ascii=False).encode()
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=self.valves.SCORE_TIMEOUT) as resp:
                body = json.loads(resp.read())

            content = body["choices"][0]["message"]["content"].strip()
            return self._parse_score(content)

        except Exception:
            return None

    def _try_qwen_scorer(self, user_msg: str, assistant_msg: str) -> Optional[int]:
        """Fallback: score via Qwen3.5 heuristic."""
        try:
            payload = {
                "model": "qwen3.5-35b",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Rate this AI response on a scale of 1-5. Reply with ONLY a JSON object.\n\n"
                            "Criteria:\n"
                            "- Accuracy: Are facts correct?\n"
                            "- Completeness: Does it fully answer the question?\n"
                            "- Clarity: Is it well-structured and easy to understand?\n"
                            "- Usefulness: Would this help the user?\n\n"
                            'Format: {"score": N, "reason": "brief reason"}'
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Question: {user_msg[:1000]}\n\n"
                            f"Response: {assistant_msg[:2000]}"
                        ),
                    },
                ],
                "max_tokens": 256,
                "temperature": 0.1,
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
            }

            url = self.valves.QUALITY_SCORER_URL
            data = json.dumps(payload, ensure_ascii=False).encode()
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=self.valves.SCORE_TIMEOUT) as resp:
                body = json.loads(resp.read())

            content = body["choices"][0]["message"]["content"].strip()
            return self._parse_score(content)

        except Exception:
            return None

    def _parse_score(self, content: str) -> Optional[int]:
        """Extract score (1-5) from scorer response."""
        import re

        # Try JSON parse
        try:
            obj = json.loads(content)
            score = int(obj.get("score", 0))
            if 1 <= score <= 5:
                return score
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        # Try regex extraction
        match = re.search(r'"score"\s*:\s*(\d)', content)
        if match:
            score = int(match.group(1))
            if 1 <= score <= 5:
                return score

        # Try bare number
        match = re.search(r'\b([1-5])\b', content)
        if match:
            return int(match.group(1))

        return None

    def _log_score(
        self, user_msg: str, assistant_msg: str, score: int, body: dict
    ) -> None:
        """Append score to the JSONL log file."""
        try:
            import hashlib

            prompt_hash = hashlib.sha256(user_msg.encode()).hexdigest()[:16]
            model = body.get("model", "unknown")

            entry = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "prompt_hash": prompt_hash,
                "model": model,
                "score": score,
                "prompt_len": len(user_msg),
                "response_len": len(assistant_msg),
                "prompt_preview": user_msg[:100],
            }

            log_path = self.valves.LOG_PATH
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        except Exception:
            # Logging failure should never break the response
            pass

    def _score_to_badge(self, score: int) -> str:
        """Convert score to a small visual badge."""
        badges = {
            1: "Quality: 1/5",
            2: "Quality: 2/5",
            3: "Quality: 3/5",
            4: "Quality: 4/5",
            5: "Quality: 5/5",
        }
        return f"*{badges.get(score, '')}*"
