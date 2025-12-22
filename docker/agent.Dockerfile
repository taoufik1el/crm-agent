# Dockerfile for building the agent service
FROM python:3.12-slim

# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml ./
COPY src ./src

# Install deps
RUN pip install --upgrade pip \
    && pip install .[agent]

ENV APP_HOST=0.0.0.0
ENV APP_PORT=8001

# Expose agent port
EXPOSE 8001

# Run agent API
CMD ["python", "-m", "uvicorn", "src.agent.api:app", "--host", "0.0.0.0", "--port", "8001"]
