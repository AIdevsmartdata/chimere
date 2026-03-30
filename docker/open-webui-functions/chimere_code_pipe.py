"""
title: Chimere Code Mode
author: chimere-stack
version: 1.0.0
description: Pipe function that creates a "chimere-code" model in Open WebUI. Routes requests through ODO's code pipeline with structured reasoning, search context, and step-by-step plans displayed as collapsible Markdown sections.
license: Apache-2.0
"""

import json
import urllib.request
import urllib.error
import re
from typing import Generator, Iterator, Optional, Union
from pydantic import BaseModel, Field


class Pipe:
    """Open WebUI Pipe that exposes a 'chimere-code' virtual model.

    When selected, requests are routed through the ODO code pipeline which:
    1. Classifies the intent and selects the 'code' route
    2. Enriches with search results if needed
    3. Applies code-optimized sampling (temp 0.6, think mode)
    4. Returns the response with structured metadata sections
    """

    class Valves(BaseModel):
        ODO_BASE_URL: str = Field(
            default="http://odo:8084",
            description="Base URL of the ODO orchestrator",
        )
        NOTHINK_URL: str = Field(
            default="http://odo:8086",
            description="Base URL of the no-think proxy (for fast generation)",
        )
        TIMEOUT_SECONDS: int = Field(
            default=300,
            description="HTTP timeout for code generation requests",
        )
        ENABLE_RESEARCH: bool = Field(
            default=True,
            description="Enable automatic web search for code questions",
        )
        ENABLE_PLANNING: bool = Field(
            default=True,
            description="Enable step-by-step plan generation before coding",
        )
        SHOW_REASONING: bool = Field(
            default=True,
            description="Show the model's reasoning/thinking in a collapsible section",
        )
        MAX_TOKENS: int = Field(
            default=16384,
            description="Maximum tokens for code generation",
        )

    def __init__(self):
        self.valves = self.Valves()

    def pipes(self) -> list[dict]:
        """Register the chimere-code virtual model."""
        return [
            {
                "id": "chimere-code",
                "name": "Chimere Code",
            }
        ]

    async def pipe(self, body: dict) -> Union[str, Generator]:
        """Process a code request through the Chimere stack.

        Enriches the response with collapsible metadata sections:
        - Research: web search results used as context
        - Reasoning: the model's thinking process
        - Plan: step-by-step implementation plan
        """

        messages = body.get("messages", [])
        stream = body.get("stream", False)

        if not messages:
            return "No messages provided."

        user_text = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_text = msg.get("content", "")
                break

        if not user_text:
            return "No user message found."

        # Build the enriched response
        sections = {}

        # Step 1: Research (optional)
        if self.valves.ENABLE_RESEARCH and self._needs_research(user_text):
            search_context = self._do_research(user_text)
            if search_context:
                sections["research"] = search_context

        # Step 2: Planning (optional)
        plan = None
        if self.valves.ENABLE_PLANNING:
            plan = self._generate_plan(user_text, sections.get("research", ""))
            if plan:
                sections["plan"] = plan

        # Step 3: Main code generation via ODO
        payload = self._build_payload(body, sections)

        if stream:
            return self._stream_response(payload, sections)
        else:
            return self._sync_response(payload, sections)

    def _needs_research(self, text: str) -> bool:
        """Heuristic: does this code question benefit from web search?"""
        research_signals = [
            r"\b(?:how to|comment|library|framework|package|api|sdk)\b",
            r"\b(?:latest|newest|current|2025|2026|version)\b",
            r"\b(?:best practice|recommended|standard|convention)\b",
            r"\b(?:error|bug|issue|fix|solve|debug|traceback)\b",
            r"\b(?:documentation|docs|reference|example)\b",
        ]
        for pattern in research_signals:
            if re.search(pattern, text, re.I):
                return True
        return False

    def _do_research(self, query: str) -> Optional[str]:
        """Call ODO search for code-relevant context."""
        try:
            payload = {
                "model": "qwen3.5-35b",
                "messages": [
                    {"role": "user", "content": f"/research {query}"},
                ],
                "max_tokens": 4096,
                "stream": False,
                "chimere_search": {
                    "depth": "quick",
                    "domain": "code",
                    "return_context_only": True,
                },
            }
            url = f"{self.valves.ODO_BASE_URL}/v1/search"
            data = json.dumps(payload, ensure_ascii=False).encode()
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read())

            sources = result.get("sources", [])
            if not sources:
                return None

            parts = []
            for i, src in enumerate(sources[:5], 1):
                title = src.get("title", "")
                snippet = src.get("snippet", "")
                url = src.get("url", "")
                parts.append(f"{i}. **{title}** ({url})\n   {snippet}")
            return "\n".join(parts)

        except Exception:
            return None

    def _generate_plan(self, user_text: str, research_context: str) -> Optional[str]:
        """Generate a step-by-step plan using the no-think proxy (fast)."""
        try:
            context_hint = ""
            if research_context:
                context_hint = f"\n\nRelevant context:\n{research_context[:2000]}"

            payload = {
                "model": "qwen3.5-35b",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a code planner. Given a coding task, output ONLY a concise "
                            "step-by-step plan as a Markdown checklist. No code, no explanation. "
                            "Maximum 8 steps. Each step should be one sentence."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"{user_text}{context_hint}",
                    },
                ],
                "max_tokens": 1024,
                "temperature": 0.3,
                "stream": False,
            }

            url = f"{self.valves.NOTHINK_URL}/v1/chat/completions"
            data = json.dumps(payload, ensure_ascii=False).encode()
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read())

            content = body["choices"][0]["message"]["content"].strip()
            if content:
                return content
        except Exception:
            pass
        return None

    def _build_payload(self, original_body: dict, sections: dict) -> dict:
        """Build the payload for the main code generation call."""
        messages = list(original_body.get("messages", []))

        # Inject research context into system prompt if available
        research = sections.get("research", "")
        plan = sections.get("plan", "")

        system_injection = []
        if research:
            system_injection.append(
                f"<search_context>\n{research}\n</search_context>"
            )
        if plan:
            system_injection.append(
                f"<plan>\n{plan}\n</plan>"
            )

        if system_injection:
            injection = "\n\n".join(system_injection)
            # Prepend to existing system message or create one
            if messages and messages[0].get("role") == "system":
                messages[0] = dict(messages[0])
                messages[0]["content"] = (
                    messages[0]["content"] + "\n\n" + injection
                )
            else:
                messages.insert(
                    0,
                    {"role": "system", "content": injection},
                )

        return {
            "model": "qwen3.5-35b",
            "messages": messages,
            "max_tokens": self.valves.MAX_TOKENS,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "stream": False,
        }

    def _sync_response(self, payload: dict, sections: dict) -> str:
        """Non-streaming code generation."""
        url = f"{self.valves.ODO_BASE_URL}/v1/chat/completions"
        data = json.dumps(payload, ensure_ascii=False).encode()
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                req, timeout=self.valves.TIMEOUT_SECONDS
            ) as resp:
                body = json.loads(resp.read())
        except Exception as e:
            return f"Error calling ODO: {e}"

        content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
        reasoning_content = (
            body.get("choices", [{}])[0]
            .get("message", {})
            .get("reasoning_content", "")
        )

        return self._assemble_response(content, reasoning_content, sections)

    def _stream_response(self, payload: dict, sections: dict) -> Generator:
        """Streaming code generation.

        Yields the collapsible sections first, then streams the main response.
        """
        # Yield pre-computed sections first
        header = self._build_sections_header(sections)
        if header:
            yield header

        # Then stream the main response from ODO
        payload["stream"] = True
        url = f"{self.valves.ODO_BASE_URL}/v1/chat/completions"
        data = json.dumps(payload, ensure_ascii=False).encode()
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            resp = urllib.request.urlopen(
                req, timeout=self.valves.TIMEOUT_SECONDS
            )
            for line in resp:
                line = line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data: "):
                    continue
                chunk_str = line[6:]
                if chunk_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(chunk_str)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    text = delta.get("content", "")
                    if text:
                        yield text
                except json.JSONDecodeError:
                    continue
            resp.close()
        except Exception as e:
            yield f"\n\n[Streaming error: {e}]"

    def _assemble_response(
        self, content: str, reasoning: str, sections: dict
    ) -> str:
        """Assemble the final response with collapsible sections."""
        parts = []

        header = self._build_sections_header(sections, reasoning)
        if header:
            parts.append(header)

        parts.append(content)
        return "\n\n".join(parts)

    def _build_sections_header(
        self, sections: dict, reasoning: str = ""
    ) -> str:
        """Build the collapsible Markdown sections."""
        parts = []

        research = sections.get("research", "")
        if research:
            n_sources = research.count("\n") + 1
            parts.append(
                f"<details><summary>Research ({n_sources} sources)</summary>\n\n"
                f"{research}\n\n</details>"
            )

        if reasoning and self.valves.SHOW_REASONING:
            # Clean up thinking tags
            clean = re.sub(r"</?think>", "", reasoning).strip()
            if clean:
                parts.append(
                    f"<details><summary>Reasoning</summary>\n\n"
                    f"{clean}\n\n</details>"
                )

        plan = sections.get("plan", "")
        if plan:
            parts.append(
                f"<details><summary>Plan</summary>\n\n"
                f"{plan}\n\n</details>"
            )

        return "\n\n".join(parts)
