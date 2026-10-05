# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Run as an unprivileged user.
RUN groupadd --system arcane && useradd --system --gid arcane --home-dir /app arcane

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY main.py ./
COPY arcane ./arcane

RUN mkdir -p /app/data /app/logs && chown -R arcane:arcane /app
USER arcane

# SQLite database and logs live here; mount volumes to keep them.
VOLUME ["/app/data", "/app/logs"]

ENTRYPOINT ["python", "main.py"]
CMD ["run"]
