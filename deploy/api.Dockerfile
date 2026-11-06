# HinglishSense API + same-origin frontend.
#   docker build -f deploy/api.Dockerfile -t hinglishsense-api .
#   (GPU) docker build -f deploy/api.Dockerfile --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu121 -t hinglishsense-api:gpu .
# Model weights are NOT baked in: artifacts/ is mounted read-only at runtime.
FROM python:3.12.8-slim-bookworm

ARG TORCH_VERSION=2.5.1
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HINGLISH_PROJECT_ROOT=/app \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HOME=/app/state/hf-cache

WORKDIR /app
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --index-url ${TORCH_INDEX} torch==${TORCH_VERSION} \
 && pip install -r backend/requirements.txt

# Source: shared research package, configs, API, frontend (no data, no weights).
COPY configs ./configs
COPY python/src ./python/src
COPY backend ./backend
COPY frontend ./frontend

RUN useradd --create-home --uid 10001 hs \
 && mkdir -p /app/state /app/artifacts /app/data/processed \
 && chown -R hs:hs /app/state
USER hs

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=4).status==200 else 1)"

CMD ["uvicorn", "app.main:create_app", "--factory", "--app-dir", "backend", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
