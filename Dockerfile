FROM python:3.12-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --frozen

COPY backend/ backend/
COPY frontend/ frontend/
COPY models/ models/
COPY docs/ docs/
COPY scripts/ scripts/

ENV PYTHONPATH=/app
ENV HISTORY_DB_PATH=/app/data/history.db

RUN mkdir -p /app/data

EXPOSE 8003 8501

CMD ["uv", "run", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8003"]
