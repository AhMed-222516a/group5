# =============================================================================
# Stage 1 – dependency installation
# =============================================================================
FROM python:3.12-slim AS builder

# Keeps Python output unbuffered and suppresses .pyc files during the build.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install OS-level build dependencies required by some Python packages
# (e.g. faiss-cpu, tokenizers) and clean up the apt cache in the same layer.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements file first so Docker can cache this expensive
# layer independently of application-code changes.
COPY requirements-deployment.txt .

# Upgrade pip, then install all dependencies.
# faiss-cpu is the CPU-compatible FAISS wheel required by the retrieval layer;
# it is not listed in requirements.txt (which references the library by use,
# not by wheel name) so we install it explicitly here.
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements-deployment.txt


# =============================================================================
# Stage 2 – production image
# =============================================================================
FROM python:3.12-slim AS production

# Runtime environment variables.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Prevent HuggingFace / tokenizers from spawning additional processes
    # inside the container, which can cause instability on Azure.
    TOKENIZERS_PARALLELISM=false \
    # HuggingFace cache directory (override via docker run -e if needed).
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence_transformers \
    # Default FastAPI port – matches FastAPIConfig.port in config.py.
    PORT=8000

# Install the shared-library runtime dependency for FAISS (libgomp1) without
# the full build toolchain, keeping the final image as small as possible.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the pre-built site-packages from the builder stage instead of
# re-running pip, which avoids an expensive reinstallation in production.
COPY --from=builder /usr/local/lib/python3.12/site-packages \
                    /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy the full project after site-packages are in place so that application
# code changes do not invalidate the dependency layer.
COPY . .

# Create the cache directories with appropriate permissions so that model
# weights can be written at runtime without requiring root.
RUN mkdir -p /app/.cache/huggingface /app/.cache/sentence_transformers \
    && chmod -R 777 /app/.cache

# Create a non-root user for security hardening.  Azure App Service and
# Azure Container Apps run containers as non-root by default.
RUN addgroup --system appgroup \
    && adduser --system --ingroup appgroup --no-create-home appuser \
    && chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000

# Use the shell form so that the $PORT environment variable is expanded at
# container start time, making the port configurable via the Azure runtime
# without rebuilding the image.
CMD uvicorn deployment.app:app \
        --host 0.0.0.0 \
        --port ${PORT} \
        --workers 1 \
        --log-level info \
        --no-access-log
