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

# Determine effective Ollama URL for the preflight check.
OLLAMA_HOST="${OLLAMA_HOST:-http://host.docker.internal:11434}"

preflight_check() {
    if ! command -v curl >/dev/null 2>&1; then
        return 0
    fi

    # On macOS, Ollama normally listens on localhost:11434.  The *container*
    # uses host.docker.internal, but from the host shell localhost is the right
    # address to preflight.  Try both.
    reachable=false
    for url in "http://localhost:11434" "$OLLAMA_HOST"; do
        if curl -fsS "$url/api/tags" >/dev/null 2>&1; then
            reachable=true
            break
        fi
    done

    if [[ "$reachable" != true ]]; then
        echo "⚠️  Could not reach Ollama from this host." >&2
        echo "    The benchmark container will use $OLLAMA_HOST." >&2
        echo "" >&2
        if [[ "$OSTYPE" == darwin* ]]; then
            echo "    Make sure the Ollama app is running (e.g. \`ollama pull gemma4:e2b\`)." >&2
            echo "    Docker Desktop will proxy host.docker.internal to the host." >&2
        else
            echo "    Linux: Ollama must listen on an interface reachable from Docker." >&2
            echo "    Start it with: OLLAMA_HOST=0.0.0.0:11434 ollama serve" >&2
            echo "    Or override:   OLLAMA_HOST=http://<host-ip>:11434 scripts/docker-benchmark.sh ..." >&2
        fi
        echo "" >&2
        read -r -p "Continue anyway? [y/N] " answer >&2 || true
        [[ "$answer" =~ ^[Yy]$ ]] || exit 1
    fi
}

cd "$REPO_ROOT"

COMMAND="${1:-local}"
shift || true

# Skip preflight for build/shell where a model isn't required.
case "$COMMAND" in
    local|bench|benchmark|jinx|jinxes|rate|compare)
        preflight_check
        ;;
esac

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
