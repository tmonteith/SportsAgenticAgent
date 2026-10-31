# Build Stage
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip wheel --no-cache-dir --wheel-dir /build/wheels -r requirements.txt

# Runtime Stage
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

# Create non-root user
RUN groupadd --system appgroup && \
    useradd --system --gid appgroup --home-dir /app --no-create-home appuser

# Install pre-built wheels from builder stage
COPY --from=builder /build/wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels /wheels/* && \
    rm -rf /wheels

# Copy application source and agent skills
COPY app ./app
COPY skills ./skills
COPY .agents/skills ./.agents/skills

RUN chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
