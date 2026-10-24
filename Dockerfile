# Offline pipeline image (CPU). Licensed data is mounted, never baked in:
#   docker build -t recall . && docker run --rm -v $PWD/data:/app/data recall make test
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends make git && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt requirements-ci.txt ./
RUN pip install --no-cache-dir -r requirements-ci.txt open_clip_torch transformers opencv-python-headless
COPY . .
ENV PYTHONPATH=/app PY=python
CMD ["make", "test"]
