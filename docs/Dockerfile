# ---- Build stage ----
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build dependencies
RUN pip install --no-cache-dir build

# Copy project files
COPY pyproject.toml README.md ./
COPY src/ src/

# Build wheel
RUN python -m build --wheel --outdir /build/dist

# ---- Runtime stage ----
FROM python:3.12-slim AS runtime

LABEL maintainer="AgentGate Contributors"
LABEL org.opencontainers.image.source="https://github.com/agentgate/agentgate"
LABEL org.opencontainers.image.description="Pre-deployment quality gate for AI agents"
LABEL org.opencontainers.image.licenses="Apache-2.0"

# Create non-root user
RUN groupadd --gid 1000 agentgate && \
    useradd --uid 1000 --gid agentgate --create-home agentgate

WORKDIR /app

# Copy and install the built wheel
COPY --from=builder /build/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && \
    rm -rf /tmp/*.whl

# Pre-download the default sentence-transformers model so it's cached in the image
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Switch to non-root user
USER agentgate

# Default: show help
ENTRYPOINT ["agentgate"]
CMD ["--help"]
