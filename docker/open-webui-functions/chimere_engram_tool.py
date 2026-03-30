"""
title: Chimere Engram Lookup
author: chimere-stack
version: 1.0.0
description: Tool that allows the LLM to query Chimere's Engram tables for domain-specific knowledge. Engram stores validated n-gram patterns from high-quality responses, providing a local knowledge base for specialized domains (medical, cyber, code).
license: Apache-2.0
"""

import json
import urllib.request
import urllib.error
from typing import Optional
from pydantic import BaseModel, Field


class Tools:
    """Open WebUI Tool for querying the Chimere Engram knowledge store.

    Engram is a specialized n-gram knowledge base that stores validated
    patterns from high-quality model responses. It supports:
    - Exact hash lookup (fast, O(1))
    - Semantic FAISS-based fuzzy matching (top-K by cosine similarity)
    - Domain-filtered queries (kine, cyber, code, general)

    The LLM can call this tool when it needs domain-specific knowledge
    that may not be in its training data.
    """

    class Valves(BaseModel):
        ODO_BASE_URL: str = Field(
            default="http://odo:8084",
            description="Base URL of the ODO orchestrator",
        )
        ENGRAM_ENDPOINT: str = Field(
            default="http://odo:8084/v1/engram/query",
            description="Direct endpoint for Engram queries (if exposed by ODO)",
        )
        EMBEDDING_URL: str = Field(
            default="http://odo:8081/v1/embeddings",
            description="Embeddings endpoint for semantic search",
        )
        DEFAULT_TOP_K: int = Field(
            default=5,
            description="Default number of results for semantic queries",
        )
        MIN_SIMILARITY: float = Field(
            default=0.6,
            description="Minimum cosine similarity threshold for results",
        )
        TIMEOUT_SECONDS: int = Field(
            default=15,
            description="HTTP timeout for Engram queries",
        )

    def __init__(self):
        self.valves = self.Valves()

    async def engram_lookup(
        self,
        query: str,
        domain: str = "auto",
        __event_emitter__=None,
    ) -> str:
        """Query the Chimere Engram knowledge base for domain-specific information.

        Engram contains validated knowledge patterns from previous high-quality
        responses. Use this when you need specialized domain knowledge about
        topics like physiotherapy/kinesiology, cybersecurity, or programming
        patterns that may have been previously validated.

        :param query: The question or topic to look up in the Engram store.
        :param domain: Domain filter: 'auto', 'kine', 'cyber', 'code', 'general'.
        :return: Matching knowledge entries from the Engram store.
        """

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Querying Engram ({domain})...",
                        "done": False,
                    },
                }
            )

        try:
            results = self._query_engram(query, domain)
        except Exception as e:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Engram query failed: {str(e)[:80]}",
                            "done": True,
                        },
                    }
                )
            return f"[Engram error: {e}]"

        entries = results.get("entries", [])

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Engram: {len(entries)} matching entries",
                        "done": True,
                    },
                }
            )

        return self._format_entries(entries, results)

    async def engram_domains(
        self,
        __event_emitter__=None,
    ) -> str:
        """List available Engram knowledge domains and their statistics.

        Use this to discover what domain knowledge is available before
        querying specific topics.

        :return: List of domains with entry counts and last-updated timestamps.
        """

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "Fetching Engram stats...",
                        "done": False,
                    },
                }
            )

        try:
            stats = self._get_stats()
        except Exception as e:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Stats failed: {str(e)[:80]}",
                            "done": True,
                        },
                    }
                )
            return f"[Engram stats error: {e}]"

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "Engram stats loaded",
                        "done": True,
                    },
                }
            )

        return self._format_stats(stats)

    def _query_engram(self, query: str, domain: str) -> dict:
        """Call the Engram query endpoint."""

        payload = {
            "query": query,
            "domain": domain,
            "top_k": self.valves.DEFAULT_TOP_K,
            "min_similarity": self.valves.MIN_SIMILARITY,
        }

        url = self.valves.ENGRAM_ENDPOINT
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
                return json.loads(resp.read())
        except urllib.error.HTTPError:
            # Fallback: try semantic search via the embeddings approach
            return self._semantic_fallback(query, domain)

    def _semantic_fallback(self, query: str, domain: str) -> dict:
        """Fallback: query ODO with an engram-hint in the payload."""

        payload = {
            "model": "qwen3.5-35b",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a knowledge retrieval assistant. "
                        "Search the Engram knowledge base and return relevant entries."
                    ),
                },
                {"role": "user", "content": query},
            ],
            "max_tokens": 2048,
            "stream": False,
            "chimere_engram": {
                "domain": domain,
                "top_k": self.valves.DEFAULT_TOP_K,
            },
        }

        url = f"{self.valves.ODO_BASE_URL}/v1/chat/completions"
        data = json.dumps(payload, ensure_ascii=False).encode()
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(
            req, timeout=self.valves.TIMEOUT_SECONDS
        ) as resp:
            body = json.loads(resp.read())

        content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
        engram_info = body.get("chimere_engram_info", {})

        entries = engram_info.get("entries", [])
        if not entries and content:
            # Parse content as a single entry
            entries = [{"text": content, "domain": domain, "similarity": 1.0}]

        return {"entries": entries, "method": "fallback"}

    def _get_stats(self) -> dict:
        """Get Engram statistics."""
        url = f"{self.valves.ODO_BASE_URL}/stats"
        req = urllib.request.Request(url, method="GET")

        with urllib.request.urlopen(
            req, timeout=self.valves.TIMEOUT_SECONDS
        ) as resp:
            stats = json.loads(resp.read())

        return stats.get("engram", {})

    def _format_entries(self, entries: list, meta: dict) -> str:
        """Format Engram entries as readable context."""

        if not entries:
            return "[No matching Engram entries found for this query.]"

        method = meta.get("method", "semantic")
        parts = [f"## Engram Knowledge ({len(entries)} entries, {method} match)\n"]

        for i, entry in enumerate(entries, 1):
            text = entry.get("text", entry.get("response", ""))
            domain = entry.get("domain", "general")
            similarity = entry.get("similarity", entry.get("score", 0))
            prompt_hash = entry.get("prompt_hash", "")

            sim_str = f" ({similarity:.0%} match)" if similarity else ""
            hash_str = f" [{prompt_hash[:8]}]" if prompt_hash else ""

            parts.append(f"### Entry {i} [{domain}]{sim_str}{hash_str}")
            parts.append(f"{text}\n")

        return "\n".join(parts)

    def _format_stats(self, stats: dict) -> str:
        """Format Engram statistics."""

        if not stats:
            return "[Engram statistics not available. The service may not expose stats.]"

        parts = ["## Engram Knowledge Base Statistics\n"]

        total = stats.get("total_entries", "?")
        parts.append(f"**Total entries:** {total}")

        domains = stats.get("domains", {})
        if domains:
            parts.append("\n**Domains:**")
            for domain, count in sorted(domains.items()):
                parts.append(f"- **{domain}**: {count} entries")

        semantic = stats.get("semantic", {})
        if semantic:
            parts.append(f"\n**Semantic index:** {semantic.get('indexed', '?')} vectors")
            parts.append(f"**Embedding dim:** {semantic.get('dim', '?')}")

        last_update = stats.get("last_update", "")
        if last_update:
            parts.append(f"\n**Last updated:** {last_update}")

        return "\n".join(parts)
