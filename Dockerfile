FROM python:3.12-slim

LABEL maintainer="Chris Agostino <info@npcworldwi.de>"
LABEL description="npcsh — AI-powered command-line shell with Rust kernel"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsqlite3-dev \
    libssl-dev \
    pkg-config \
    curl \
    git \
    espeak \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Rust and npcrs from crates.io
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:$PATH"
RUN cargo install npcrs

WORKDIR /app

COPY setup.py README.md ./
COPY npcsh/ npcsh/

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e ".[lite]"

RUN mkdir -p /root/.npcsh/npc_team/jinxes /data

ENV NPCSH_DB_PATH=/data/npcsh_history.db
ENV NPCSH_BASE=/root/.npcsh

VOLUME ["/data", "/root/.npcsh", "/workspace"]
WORKDIR /workspace

ENTRYPOINT ["npcsh"]
