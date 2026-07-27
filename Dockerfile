# FastAPI-only container for the Delivery Risk Intelligence API.
#
# python:3.11-slim keeps the OS layer smaller than the full Python image while
# retaining compatible wheels for pandas, NumPy, and scikit-learn.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Dependency metadata is copied before source so application-only changes can
# reuse Docker's expensive package-install cache.
COPY requirements-api.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-api.txt

# The public service runs as an unprivileged account rather than root.
RUN useradd --create-home --uid 10001 appuser

# Copy only files imported by the production API. Training, evaluation,
# notebooks, reports, Streamlit, raw data, and experiment artifacts are absent.
COPY --chown=appuser:appuser src/__init__.py src/observability.py ./src/
COPY --chown=appuser:appuser \
    src/api/__init__.py \
    src/api/config.py \
    src/api/cors.py \
    src/api/errors.py \
    src/api/logging_config.py \
    src/api/main.py \
    src/api/rate_limit.py \
    src/api/readiness.py \
    src/api/request_context.py \
    src/api/security.py \
    ./src/api/
COPY --chown=appuser:appuser \
    src/agent/__init__.py \
    src/agent/tools.py \
    src/agent/router.py \
    src/agent/langgraph_agent.py \
    ./src/agent/
COPY --chown=appuser:appuser \
    src/features/__init__.py \
    src/features/feature_contract.py \
    ./src/features/
# ADD automatically extracts this checksum-documented archive into its original
# models/ and data/processed/ paths. It contains no credentials or raw data.
ADD --chown=appuser:appuser artifacts/runtime_artifacts.tar.gz /app/

USER appuser

EXPOSE 8000

# Readiness confirms required model artifacts without calling an external LLM.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=3)"]

# 0.0.0.0 is required so platform traffic reaches Uvicorn. Cloud Run injects
# PORT; local Docker falls back to 8000.
CMD ["sh", "-c", "exec uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
