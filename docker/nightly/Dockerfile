# =============================================================================
# Chimere Nightly Self-Improvement Container
# Runs overnight LoRA training, Engram updates, and DSPy prompt optimization
# on a configurable schedule.
#
# Build:
#   docker build -t chimere-nightly -f docker/nightly/Dockerfile .
#
# Run:
#   docker run -d \
#     -v chimere-logs:/data/logs \
#     -v chimere-engram:/data/engram \
#     -v chimere-soul:/data/soul \
#     -e ODO_BACKEND=http://inference:8081 \
#     chimere-nightly
# =============================================================================

FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies for nightly jobs
RUN pip install --no-cache-dir \
    requests \
    pyyaml

# Create data directories
RUN mkdir -p \
    /data/logs/nightly \
    /data/engram \
    /data/soul \
    /app/scripts

# Copy nightly scripts from chimere-odo
COPY chimere-odo/quality/nightly_lora.py       /app/scripts/nightly_lora.py
COPY chimere-odo/engram/engram_write_nightly.py /app/scripts/engram_write_nightly.py
COPY chimere-odo/quality/dspy_optimize.py      /app/scripts/dspy_optimize.py
COPY chimere-odo/quality/lora_mezo_nightly.py  /app/scripts/lora_mezo_nightly.py

# Copy the scheduler
COPY docker/nightly/scheduler.py /app/scheduler.py

# ---------------------------------------------------------------------------
# Schedule configuration
# ---------------------------------------------------------------------------
# NIGHTLY_SCHEDULE: time to run jobs (HH:MM in 24h format, local container TZ)
# NIGHTLY_LORA:     enable LoRA training from quality-validated response pairs
# NIGHTLY_ENGRAM:   enable Engram n-gram table updates from scored responses
# NIGHTLY_DSPY:     enable DSPy MIPROv2 system-prompt optimization
# NIGHTLY_MEZO:     enable MeZO zeroth-order LoRA (fits 16 GB, no backward pass)
# ODO_BACKEND:      inference server URL (used by scripts for scoring/eval)
# ---------------------------------------------------------------------------
ENV NIGHTLY_SCHEDULE=00:30 \
    NIGHTLY_LORA=true \
    NIGHTLY_ENGRAM=true \
    NIGHTLY_DSPY=false \
    NIGHTLY_MEZO=false \
    ODO_BACKEND=http://inference:8081 \
    PYTHONUNBUFFERED=1

# Log and data volumes
VOLUME ["/data/logs", "/data/engram", "/data/soul"]

WORKDIR /app

HEALTHCHECK --interval=60s --timeout=5s --retries=3 \
    CMD python3 -c "import os, time; s=os.path.getmtime('/tmp/scheduler-heartbeat') if os.path.exists('/tmp/scheduler-heartbeat') else 0; exit(0 if time.time()-s<120 else 1)"

ENTRYPOINT ["python3", "/app/scheduler.py"]
