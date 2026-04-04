FROM rust:latest AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config libsqlite3-dev libssl-dev libclang-dev \
    && rm -rf /var/lib/apt/lists/*

RUN cargo install npcrs

FROM debian:bookworm-slim

LABEL maintainer="Chris Agostino <info@npcworldwi.de>"
LABEL description="npcsh — AI-powered command-line shell (Rust)"

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsqlite3-0 libssl3 ca-certificates curl git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/cargo/bin/npcrs /usr/local/bin/npcsh

RUN mkdir -p /root/.npcsh/npc_team/jinxes /data

COPY npcsh/npc_team/ /root/.npcsh/npc_team/

ENV NPCSH_DB_PATH=/data/npcsh_history.db
ENV NPCSH_BASE=/root/.npcsh

VOLUME ["/data", "/root/.npcsh", "/workspace"]
WORKDIR /workspace

ENTRYPOINT ["npcsh"]
