#!/bin/bash
set -e

# Docker entrypoint for npcsh benchmark runs.
# Keeps all state under /data/npcsh so the host's ~/.npcsh is never touched.

# Ensure the data volume directories exist.
mkdir -p /data/npcsh/benchmarks/local \
         /data/npcsh/benchmarks/jinxes \
         /data/npcsh/benchmarks/ratings \
         /data/npcsh/benchmarks/datasets \
         /workspace

# Always re-sync the bundled npc_team into the pre-seeded home so jinx changes
# in the repo are reflected in the container without copying from the host.
if [ -d /opt/npcsh/npcsh/npc_team ]; then
    mkdir -p /root/.npcsh/npc_team
    cp -r /opt/npcsh/npcsh/npc_team/* /root/.npcsh/npc_team/ 2>/dev/null || true
fi

# Keep the Rust binary from trying to write a new rc file if /init is run.
export NPCSH_INITIALIZED=1

COMMAND="${1:-local}"
shift || true

case "$COMMAND" in
    local|bench|benchmark)
        exec python -m npcsh.benchmark.local_runner "$@"
        ;;
    jinx|jinxes)
        exec python -m npcsh.jinx_tester "$@"
        ;;
    rate)
        exec python scripts/rate_traces.py "$@"
        ;;
    compare)
        exec python scripts/compare_benchmarks.py "$@"
        ;;
    shell|bash|sh)
        exec /bin/bash "$@"
        ;;
    python|py)
        exec python "$@"
        ;;
    npcsh)
        exec npcsh "$@"
        ;;
    *)
        exec "$COMMAND" "$@"
        ;;
esac
