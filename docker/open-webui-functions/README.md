# Chimere Stack -- Open WebUI Integration Functions

Plugin functions for integrating the Chimere Stack (ODO orchestrator, deep search,
Engram knowledge base, quality gate) with Open WebUI.

## Files

| File | Type | Description |
|------|------|-------------|
| `chimere_search_tool.py` | **Tool** | Deep web search via SOTA 2026 pipeline (SearXNG + Brave + ChromaDB + neural reranking) |
| `chimere_code_pipe.py` | **Pipe Function** | "Chimere Code" virtual model with research, reasoning, and planning sections |
| `chimere_engram_tool.py` | **Tool** | Engram knowledge base lookup (semantic + hash-based n-gram retrieval) |
| `chimere_quality_filter.py` | **Filter Function** | Post-response quality scoring with ThinkPRM/Qwen3.5, JSONL logging |

## Installation

### Method 1: Import via Admin Panel (recommended)

1. Open your Open WebUI instance (e.g., `http://localhost:3000`)
2. Go to **Workspace > Functions** (admin access required)
3. Click the **+** button to create a new function
4. Copy-paste the contents of each `.py` file into the editor
5. Save and enable each function

### Method 2: Import from file

1. Go to **Workspace > Functions**
2. Click the **Import** button (upload icon)
3. Select the `.py` file to import
4. Repeat for each function

### Method 3: API import

```bash
# Replace HOST with your Open WebUI URL and TOKEN with your API key
HOST="http://localhost:3000"
TOKEN="your-api-key"

for f in chimere_search_tool.py chimere_code_pipe.py chimere_engram_tool.py chimere_quality_filter.py; do
    curl -X POST "$HOST/api/v1/functions/create" \
        -H "Authorization: Bearer $TOKEN" \
        -H "Content-Type: application/json" \
        -d "{\"id\": \"$(basename $f .py)\", \"name\": \"$(head -3 $f | grep 'title:' | cut -d: -f2 | xargs)\", \"content\": $(python3 -c "import json,sys; print(json.dumps(open('$f').read()))")}"
done
```

## Configuration

After installing, configure each function's **Valves** (settings) via the
Open WebUI admin panel:

### chimere_search_tool

| Valve | Default | Description |
|-------|---------|-------------|
| `ODO_BASE_URL` | `http://odo:8084` | ODO orchestrator URL. Use `http://127.0.0.1:8084` if running on host. |
| `SEARCH_DEPTH` | `standard` | `quick` (~20s), `standard` (~30s), `deep` (~60s) |
| `SEARCH_DOMAIN` | `auto` | `auto`, `medical`, `cyber`, `code`, `general` |
| `TIMEOUT_SECONDS` | `120` | HTTP request timeout |
| `MAX_RESULTS` | `8` | Maximum search results to return |

### chimere_code_pipe

| Valve | Default | Description |
|-------|---------|-------------|
| `ODO_BASE_URL` | `http://odo:8084` | ODO orchestrator URL |
| `NOTHINK_URL` | `http://odo:8086` | No-think proxy URL (fast generation for plans) |
| `ENABLE_RESEARCH` | `true` | Auto web search for code questions |
| `ENABLE_PLANNING` | `true` | Generate step-by-step plan before coding |
| `SHOW_REASONING` | `true` | Show reasoning in collapsible section |
| `MAX_TOKENS` | `16384` | Max tokens for code generation |

### chimere_engram_tool

| Valve | Default | Description |
|-------|---------|-------------|
| `ODO_BASE_URL` | `http://odo:8084` | ODO orchestrator URL |
| `ENGRAM_ENDPOINT` | `http://odo:8084/v1/engram/query` | Direct Engram query endpoint |
| `DEFAULT_TOP_K` | `5` | Number of results for semantic queries |
| `MIN_SIMILARITY` | `0.6` | Minimum cosine similarity threshold |

### chimere_quality_filter

| Valve | Default | Description |
|-------|---------|-------------|
| `QUALITY_SCORER_URL` | `http://odo:8084/v1/chat/completions` | Scorer endpoint |
| `THINKPRM_URL` | `http://odo:8085/v1/chat/completions` | ThinkPRM scorer |
| `LOG_PATH` | `/data/logs/quality_scores.jsonl` | JSONL log path |
| `LOW_SCORE_THRESHOLD` | `2` | Score at or below which warning is shown |
| `ENABLED` | `true` | Enable/disable scoring |

## Usage

### Search Tool

Enable the search tool on any model:
1. Go to **Workspace > Models**, edit your model
2. Under **Tools**, check "Chimere Deep Search"
3. The LLM will automatically use it when it needs web information
4. Users can also click the **+** button in chat to enable it per-session

### Code Mode

Select "Chimere Code" from the model dropdown. Responses will include:
- Collapsible "Research" section with relevant web sources
- Collapsible "Reasoning" section showing the model's thinking
- Collapsible "Plan" section with a step-by-step checklist
- The main code response

### Engram Lookup

Enable the Engram tool on any model. The LLM can:
- Call `engram_lookup(query, domain)` to search domain knowledge
- Call `engram_domains()` to discover available knowledge domains

### Quality Filter

Apply globally or per-model:
- **Global**: Admin Panel > Functions > Quality Gate > click globe icon
- **Per-model**: Workspace > Models > edit model > enable under Filters

Responses scoring 2/5 or below get a low-confidence indicator.
Scores are logged to `quality_scores.jsonl` for later analysis.

## Network Requirements

All functions communicate with the Chimere stack over HTTP. In a Docker setup:

```
Open WebUI  -->  ODO (8084)  -->  llama-server (8081)
                     |
                     +--> ThinkPRM (8085)
                     +--> nothink-proxy (8086)
                     +--> SearXNG (8080)
                     +--> Brave Search API
                     +--> ChromaDB
```

If Open WebUI runs in Docker alongside ODO, use the Docker service name
(e.g., `http://odo:8084`). If running on the host, use `http://127.0.0.1:8084`.

## Development

These functions run inside the Open WebUI Python environment. They use only
`urllib` for HTTP (no `requests` dependency) to avoid conflicts with Open WebUI's
package versions.

To test locally without Open WebUI:

```python
import asyncio
from chimere_search_tool import Tools

tool = Tools()
result = asyncio.run(tool.web_search("latest Python 3.13 features"))
print(result)
```
