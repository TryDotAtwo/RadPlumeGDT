ARG BASE_IMAGE=python:3.11-slim-bookworm
FROM ${BASE_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/app/.mplconfig

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY main.py LICENSE README.md PROJECT_MEMORY.md ./
COPY src ./src
COPY scripts ./scripts

RUN mkdir -p /app/data /app/outputs /app/.mplconfig

VOLUME ["/app/data", "/app/outputs"]
ENTRYPOINT ["python", "main.py"]
CMD ["report"]
