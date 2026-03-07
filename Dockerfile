#Stage 1
FROM python:3.13-slim AS builder

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev

#Stage 2
FROM python:3.13-slim AS runtime

WORKDIR /app

COPY --from=builder /app/.venv .venv

COPY src/ ./src/
COPY data/ ./data/

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000
EXPOSE 8080
EXPOSE 8001