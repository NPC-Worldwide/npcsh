FROM debian:trixie-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates && rm -rf /var/lib/apt/lists/*
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config libsqlite3-dev libssl-dev libclang-dev cmake \
    && rm -rf /var/lib/apt/lists/*

RUN rustup component add rustfmt && cargo install npcrs

FROM debian:trixie-slim

LABEL maintainer="Chris Agostino <info@npcworldwi.de>"
LABEL description="npcsh — AI-powered command-line shell (Rust)"

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsqlite3-0 libssl3 libgomp1 ca-certificates curl git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/cargo/bin/npcrs /usr/local/bin/npcsh

RUN mkdir -p /root/.npcsh/npc_team/jinxes /data

COPY npcsh/npc_team/ /root/.npcsh/npc_team/

ENV NPCSH_DB_PATH=/data/npcsh_history.db
ENV NPCSH_BASE=/root/.npcsh

VOLUME ["/data", "/root/.npcsh", "/workspace"]
WORKDIR /workspace

ENTRYPOINT ["npcsh"]
