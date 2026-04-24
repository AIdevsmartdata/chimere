#!/bin/bash
# 4-request concurrent stress test against chimere-server prod (:8081)
# with full monitoring: /metrics pre/post, nvidia-smi dmon, per-req timing+content.
set -u

OUT=/tmp/stress-test
PROMPTS=$OUT/prompts.json
URL=http://127.0.0.1:8081/v1/chat/completions
MAX_TOK=8192

mkdir -p $OUT/raw

echo "=== stress test begin $(date -Iseconds) ==="

# 1) Pre-snapshot
curl -sS http://127.0.0.1:8081/metrics > $OUT/raw/metrics-pre.txt
curl -sS http://127.0.0.1:8081/v1/status > $OUT/raw/status-pre.json
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader > $OUT/raw/gpu-pre.txt

# 2) Start nvidia-smi dmon in background
nvidia-smi dmon -s pumct -d 1 -c 120 -o T > $OUT/raw/dmon.csv 2>&1 &
DMON_PID=$!
trap "kill $DMON_PID 2>/dev/null" EXIT

# 3) Launch 4 concurrent streaming requests
python3 -c "
import json, subprocess, time, os
from concurrent.futures import ThreadPoolExecutor, as_completed

prompts = json.load(open('$PROMPTS'))
out_dir = '$OUT/raw'

def drive(p, idx):
    body = {
        'model': 'chimere',
        'messages': [{'role': 'user', 'content': p['prompt']}],
        'max_tokens': $MAX_TOK,
        'stream': True
    }
    payload = json.dumps(body)
    t0 = time.monotonic()
    ttft = None
    chunks = []
    think_chunks = []
    total_toks = 0
    finish_reason = None
    result_file = f'{out_dir}/req-{idx}-{p[\"id\"]}.jsonl'
    sse_file = f'{out_dir}/req-{idx}-{p[\"id\"]}.sse'
    with open(sse_file, 'w') as sse_out:
        proc = subprocess.Popen(
            ['curl', '-sS', '-N', '--max-time', '120', '$URL',
             '-H', 'Content-Type: application/json',
             '-d', payload],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
        for line in proc.stdout:
            sse_out.write(line)
            line = line.strip()
            if not line.startswith('data: '):
                continue
            data = line[6:]
            if data == '[DONE]':
                break
            try:
                d = json.loads(data)
                delta = d.get('choices',[{}])[0].get('delta',{})
                content = delta.get('content') or ''
                reasoning = delta.get('reasoning_content') or ''
                if (content or reasoning) and ttft is None:
                    ttft = time.monotonic() - t0
                if content:
                    chunks.append(content)
                    total_toks += 1
                if reasoning:
                    think_chunks.append(reasoning)
                fr = d.get('choices',[{}])[0].get('finish_reason')
                if fr:
                    finish_reason = fr
            except Exception:
                pass
        proc.wait()
    wall = time.monotonic() - t0
    full = ''.join(chunks)
    think = ''.join(think_chunks)
    with open(result_file, 'w') as f:
        json.dump({
            'id': p['id'],
            'trap_type': p['trap_type'],
            'ttft_s': round(ttft, 3) if ttft else None,
            'wall_s': round(wall, 3),
            'content_tokens': total_toks,
            'think_chars': len(think),
            'content_chars': len(full),
            'finish_reason': finish_reason,
            'tok_per_s': round(total_toks / wall, 2) if wall > 0 else 0,
            'prompt_excerpt': p['prompt'][:120],
            'think_excerpt': think[:500],
            'content': full,
        }, f, ensure_ascii=False, indent=2)
    return p['id'], ttft, wall, total_toks, finish_reason

T0 = time.monotonic()
with ThreadPoolExecutor(max_workers=4) as pool:
    futures = [pool.submit(drive, p, i) for i, p in enumerate(prompts)]
    for fut in as_completed(futures):
        pid, ttft, wall, n, fr = fut.result()
        print(f'[done] {pid}: ttft={ttft:.2f}s wall={wall:.2f}s toks={n} finish={fr}')

T1 = time.monotonic()
print(f'=== total wall: {T1-T0:.2f}s for 4 concurrent ===')
" 2>&1 | tee $OUT/raw/driver.log

# 4) Post-snapshot
sleep 2
kill $DMON_PID 2>/dev/null
curl -sS http://127.0.0.1:8081/metrics > $OUT/raw/metrics-post.txt
curl -sS http://127.0.0.1:8081/v1/status > $OUT/raw/status-post.json
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader > $OUT/raw/gpu-post.txt

echo "=== stress test end $(date -Iseconds) ==="
echo "Raw artifacts: $OUT/raw/"
