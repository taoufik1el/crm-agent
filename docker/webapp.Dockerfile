# Dockerfile for web application
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
    && pip install .[webapp]


ENV APP_HOST=0.0.0.0
ENV APP_PORT=8001

EXPOSE 8003

CMD ["streamlit", "run", "src/webapp/app.py", "--server.port=8003", "--server.address=0.0.0.0", "--server.headless=true"]
