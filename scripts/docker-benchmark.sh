#!/bin/bash
# Run the npcsh benchmark harness inside Docker without touching host config.
# Usage:
#   scripts/docker-benchmark.sh build
#   scripts/docker-benchmark.sh local --model qwen3.5:2b --provider ollama --category shell --difficulty easy
#   scripts/docker-benchmark.sh jinx
#   scripts/docker-benchmark.sh rate --csv-dir /data/npcsh/benchmarks/local
#   scripts/docker-benchmark.sh shell
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COMPOSE_FILE="$REPO_ROOT/docker-compose.benchmark.yml"

# Results go here on the host.  It is git-ignored via the top-level results/ entry.
mkdir -p "$REPO_ROOT/results/npcsh"

cd "$REPO_ROOT"

COMMAND="${1:-local}"
shift || true

case "$COMMAND" in
    build)
        docker compose -f "$COMPOSE_FILE" build
        ;;
    local|bench|benchmark)
        docker compose -f "$COMPOSE_FILE" run --rm benchmark local "$@"
        ;;
    jinx|jinxes)
        docker compose -f "$COMPOSE_FILE" run --rm benchmark jinx "$@"
        ;;
    rate)
        docker compose -f "$COMPOSE_FILE" run --rm benchmark rate "$@"
        ;;
    compare)
        docker compose -f "$COMPOSE_FILE" run --rm benchmark compare "$@"
        ;;
    shell|bash)
        docker compose -f "$COMPOSE_FILE" run --rm benchmark shell
        ;;
    run)
        # Passthrough: run any command inside the benchmark container.
        docker compose -f "$COMPOSE_FILE" run --rm benchmark "$@"
        ;;
    *)
        echo "Usage: $(basename "$0") {build|local|jinx|rate|compare|shell|run} [args...]"
        echo ""
        echo "  build   - build the benchmark Docker image"
        echo "  local   - run npcsh.benchmark.local_runner (default)"
        echo "  jinx    - run jinx-level tests/benchmarks"
        echo "  rate    - run rate_traces.py"
        echo "  compare - run compare_benchmarks.py"
        echo "  shell   - drop into a shell in the container"
        echo "  run     - run an arbitrary command in the container"
        exit 1
        ;;
esac
