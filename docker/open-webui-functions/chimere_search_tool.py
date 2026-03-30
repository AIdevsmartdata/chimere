"""
title: Chimere Deep Search
author: chimere-stack
version: 1.0.0
description: Deep web search tool for the Chimere Stack. Calls the SOTA 2026 deep search pipeline (SearXNG + Brave + ChromaDB + neural reranking) via ODO or directly via the search endpoint.
license: MIT
"""

import json
import urllib.request
import urllib.error
from typing import Optional
from pydantic import BaseModel, Field


class Tools:
    """Open WebUI Tool that performs deep web search via the Chimere stack.

    When the LLM decides it needs web information (or the user clicks the
    search button), this tool calls the deep_search_sota pipeline through
    ODO's enrichment layer and returns formatted search results as context.
    """

    class Valves(BaseModel):
        ODO_BASE_URL: str = Field(
            default="http://odo:8084",
            description="Base URL of the ODO orchestrator (use 'odo' for Docker, '127.0.0.1' for host)",
        )
        SEARCH_DEPTH: str = Field(
            default="standard",
            description="Search depth: quick (~20s), standard (~30s), deep (~60s)",
        )
        SEARCH_DOMAIN: str = Field(
            default="auto",
            description="Domain hint: auto, medical, cyber, code, general",
        )
        TIMEOUT_SECONDS: int = Field(
            default=120,
            description="HTTP timeout for the search request in seconds",
        )
        MAX_RESULTS: int = Field(
            default=8,
            description="Maximum number of search results to return",
        )

    def __init__(self):
        self.valves = self.Valves()

    async def web_search(
        self,
        query: str,
        __event_emitter__=None,
    ) -> str:
        """Search the web using the Chimere SOTA 2026 deep search pipeline.
        Returns relevant web results with sources, ranked by relevance.

        Use this when you need up-to-date information, factual data, or
        research from the web that is beyond your training data.

        :param query: The search query to look up on the web.
        :return: Formatted search results with sources and snippets.
        """

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Searching: {query[:80]}...",
                        "done": False,
                    },
                }
            )

        try:
            results = self._call_search(query)
        except Exception as e:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Search failed: {str(e)[:100]}",
                            "done": True,
                        },
                    }
                )
            return f"[Search error: {e}]"

        if __event_emitter__:
            n = results.get("n_sources", 0)
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Found {n} sources ({results.get('search_ms', '?')}ms)",
                        "done": True,
                    },
                }
            )

            # Emit sources as citations if available
            for source in results.get("sources", []):
                await __event_emitter__(
                    {
                        "type": "citation",
                        "data": {
                            "document": [source.get("snippet", "")],
                            "metadata": [
                                {
                                    "source": source.get("url", ""),
                                    "title": source.get("title", "Unknown"),
                                }
                            ],
                            "source": {
                                "name": source.get("title", "Web"),
                                "url": source.get("url", ""),
                            },
                        },
                    }
                )

        return self._format_results(results)

    def _call_search(self, query: str) -> dict:
        """Call the search endpoint and return parsed results."""

        # Strategy 1: Call ODO with a search-enriched request
        # ODO will run deep_search_sota.py internally and return results
        payload = {
            "model": "qwen3.5-35b",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a search assistant. Return the raw search results.",
                },
                {"role": "user", "content": query},
            ],
            "max_tokens": 1,
            "stream": False,
            # Signal to ODO that this is a search-only request
            "chimere_search": {
                "depth": self.valves.SEARCH_DEPTH,
                "domain": self.valves.SEARCH_DOMAIN,
                "max_results": self.valves.MAX_RESULTS,
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

        try:
            with urllib.request.urlopen(
                req, timeout=self.valves.TIMEOUT_SECONDS
            ) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError:
            # Fallback: call deep_search_sota directly via its CLI-like endpoint
            return self._call_search_fallback(query)

    def _call_search_fallback(self, query: str) -> dict:
        """Fallback: call the search pipeline via ODO's chat completions
        with a research route hint."""

        payload = {
            "model": "qwen3.5-35b",
            "messages": [
                {
                    "role": "user",
                    "content": f"/research {query}",
                }
            ],
            "max_tokens": 4096,
            "stream": False,
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
        enrich_info = body.get("chimere_enrich_info", {})

        return {
            "synthesis": content,
            "sources": enrich_info.get("sources", []),
            "n_sources": enrich_info.get("n_sources", 0),
            "search_ms": enrich_info.get("enrich_ms", 0),
        }

    def _format_results(self, results: dict) -> str:
        """Format search results as clean context for the LLM."""

        parts = []

        # Add source snippets
        sources = results.get("sources", [])
        if sources:
            parts.append(f"## Web Search Results ({len(sources)} sources)\n")
            for i, src in enumerate(sources, 1):
                title = src.get("title", "Untitled")
                url = src.get("url", "")
                snippet = src.get("snippet", "No preview available.")
                score = src.get("score", "")
                score_str = f" (relevance: {score:.2f})" if isinstance(score, (int, float)) else ""
                parts.append(f"### [{i}] {title}{score_str}")
                parts.append(f"URL: {url}")
                parts.append(f"{snippet}\n")

        # Add synthesis if present
        synthesis = results.get("synthesis", "")
        if synthesis:
            parts.append("## Synthesis")
            parts.append(synthesis)

        if not parts:
            return "[No search results found.]"

        return "\n".join(parts)
