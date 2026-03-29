#!/bin/bash
# test_q5_selective_offload.sh — Test Q5_K_XL with selective layer offloading
#
# Strategy: Keep layers 0-29 fully on GPU, offload only layers 30-59 experts to CPU
# Expected: ~45-55 tok/s gen, ~10-30 tok/s prompt (vs 29.6/1.1 with full offload)

set -e

MODEL="$HOME/.openclaw/models/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-UD-Q5_K_XL.gguf"
PORT=8090

echo "============================================"
echo " Q5_K_XL Selective Offload Test"
echo "============================================"
echo "  Layers 0-29: GPU (experts + attention)"
echo "  Layers 30-59: CPU experts, GPU attention"
echo "  Port: $PORT"
echo ""

# Check GPU is free
GPU_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
echo "  Current GPU usage: ${GPU_USED} MiB"
if [ "$GPU_USED" -gt 1000 ]; then
    echo "  WARNING: GPU has $GPU_USED MiB in use. Need ~15.5 GB free."
    echo "  Stop other GPU processes first!"
    exit 1
fi

echo ""
echo "Starting llama-server..."

$HOME/llama.cpp/build/bin/llama-server \
    -m "$MODEL" \
    -ngl 99 \
    --flash-attn on \
    -ot "blk\.3[0-9]\.ffn_.*_exps\.weight=CPU" \
    -ot "blk\.4[0-9]\.ffn_.*_exps\.weight=CPU" \
    -ot "blk\.5[0-9]\.ffn_.*_exps\.weight=CPU" \
    --cache-type-k q8_0 \
    --cache-type-v q4_0 \
    -c 4096 \
    -np 1 \
    --host 127.0.0.1 \
    --port $PORT \
    --jinja \
    --reasoning-format deepseek \
    --no-context-shift &

SERVER_PID=$!
echo "  Server PID: $SERVER_PID"

# Wait for server
echo "  Waiting for server..."
for i in $(seq 1 60); do
    if curl -s http://127.0.0.1:$PORT/health | grep -q "ok"; then
        echo "  Server ready!"
        break
    fi
    sleep 2
done

# Check VRAM
echo ""
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
echo ""

# Test 1: Generation speed
echo "--- Test 1: Generation speed ---"
curl -s http://127.0.0.1:$PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "test",
        "messages": [{"role": "user", "content": "Write a Python function that implements merge sort with detailed comments."}],
        "max_tokens": 256,
        "temperature": 0.0,
        "stream": false,
        "chat_template_kwargs": {"enable_thinking": false}
    }' | python3 -c "
import json, sys
d = json.load(sys.stdin)
t = d.get('timings', {})
c = d['choices'][0]['message']['content']
print(f'  Gen speed:    {t.get(\"predicted_per_second\", 0):.1f} tok/s')
print(f'  Prompt speed: {t.get(\"prompt_per_second\", 0):.1f} tok/s')
print(f'  Tokens:       {t.get(\"predicted_n\", 0)}')
print(f'  Response:     {c[:100]}...')
"

# Test 2: Longer prompt eval
echo ""
echo "--- Test 2: Longer prompt for prompt eval speed ---"
curl -s http://127.0.0.1:$PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "test",
        "messages": [{"role": "user", "content": "Analyze the following code and explain every line in detail. Then rewrite it with improvements:\n\nimport asyncio\nimport aiohttp\nfrom typing import List, Dict, Optional\nimport json\nimport logging\nfrom dataclasses import dataclass, field\nfrom pathlib import Path\nimport hashlib\nimport time\n\nlogger = logging.getLogger(__name__)\n\n@dataclass\nclass CacheEntry:\n    url: str\n    content: str\n    timestamp: float\n    etag: Optional[str] = None\n    ttl: int = 3600\n\n    @property\n    def is_expired(self) -> bool:\n        return time.time() - self.timestamp > self.ttl\n\n@dataclass\nclass FetchResult:\n    url: str\n    status: int\n    content: str\n    cached: bool = False\n    error: Optional[str] = None\n\nclass AsyncWebFetcher:\n    def __init__(self, max_concurrent: int = 10, cache_dir: Optional[Path] = None):\n        self.semaphore = asyncio.Semaphore(max_concurrent)\n        self.cache: Dict[str, CacheEntry] = {}\n        self.cache_dir = cache_dir\n        if cache_dir:\n            cache_dir.mkdir(parents=True, exist_ok=True)\n\n    async def fetch_one(self, session: aiohttp.ClientSession, url: str) -> FetchResult:\n        cache_key = hashlib.md5(url.encode()).hexdigest()\n        if cache_key in self.cache and not self.cache[cache_key].is_expired:\n            return FetchResult(url=url, status=200, content=self.cache[cache_key].content, cached=True)\n        async with self.semaphore:\n            try:\n                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:\n                    content = await resp.text()\n                    self.cache[cache_key] = CacheEntry(url=url, content=content, timestamp=time.time())\n                    return FetchResult(url=url, status=resp.status, content=content)\n            except Exception as e:\n                return FetchResult(url=url, status=0, content=\"\", error=str(e))\n\n    async def fetch_all(self, urls: List[str]) -> List[FetchResult]:\n        async with aiohttp.ClientSession() as session:\n            tasks = [self.fetch_one(session, url) for url in urls]\n            return await asyncio.gather(*tasks)\n"}],
        "max_tokens": 64,
        "temperature": 0.0,
        "stream": false,
        "chat_template_kwargs": {"enable_thinking": false}
    }' | python3 -c "
import json, sys
d = json.load(sys.stdin)
t = d.get('timings', {})
print(f'  Gen speed:    {t.get(\"predicted_per_second\", 0):.1f} tok/s')
print(f'  Prompt speed: {t.get(\"prompt_per_second\", 0):.1f} tok/s')
print(f'  Prompt toks:  {t.get(\"prompt_n\", 0)}')
"

echo ""
echo "--- Cleanup ---"
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null
echo "Server stopped."
echo ""
nvidia-smi --query-gpu=memory.used --format=csv,noheader
echo "Done!"
