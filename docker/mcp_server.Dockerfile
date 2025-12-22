# Dockerfile for MCP Server
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src ./src

RUN pip install --upgrade pip \
    && pip install .[mcp_server]


ENV APP_HOST=0.0.0.0
ENV APP_PORT=8001

EXPOSE 8002

CMD ["python", "-m", "uvicorn", "src.mcp_server.server:app", "--host", "0.0.0.0", "--port", "8002"]
