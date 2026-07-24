# Running npcsh benchmarks in Docker

The benchmark harness can run inside a container so it never writes to your host's `~/.npcsh`, `~/.npcshrc`, `~/npcsh_history.db`, or shell configuration files.

## Quick start

```bash
# Build the image once.
scripts/docker-benchmark.sh build

# Run a subset of the local npcsh benchmark against a local Ollama model.
scripts/docker-benchmark.sh local --model qwen3.5:2b --provider ollama --category shell --difficulty easy

# Run jinx-level tests.
scripts/docker-benchmark.sh jinx

# Rate the resulting traces.
scripts/docker-benchmark.sh rate --csv-dir /data/npcsh/benchmarks/local
```

All outputs land in `./results/npcsh/` on the host. The container's `/data/npcsh` is the only writable state directory; your host `~/.npcsh` is never mounted.

## Networking

The container reaches Ollama at `http://host.docker.internal:11434` by default. This works automatically on Docker Desktop for macOS and Windows.

On Linux, Ollama must be listening on an interface reachable from the Docker bridge. The easiest way is:

```bash
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

Then run the benchmark normally. Alternatively, override the host address:

```bash
OLLAMA_HOST=http://<host-ip>:11434 scripts/docker-benchmark.sh local --model gemma4:e2b --provider ollama
```

## Commands

| Wrapper command | Maps to inside container |
|---|---|
| `scripts/docker-benchmark.sh build` | rebuild the image |
| `scripts/docker-benchmark.sh local ...` | `python -m npcsh.benchmark.local_runner ...` |
| `scripts/docker-benchmark.sh jinx ...` | `python -m npcsh.jinx_tester ...` |
| `scripts/docker-benchmark.sh rate ...` | `python scripts/rate_traces.py ...` |
| `scripts/docker-benchmark.sh compare ...` | `python scripts/compare_benchmarks.py ...` |
| `scripts/docker-benchmark.sh shell` | interactive shell for debugging |
| `scripts/docker-benchmark.sh run -- <cmd>` | run an arbitrary command in the container |

## Environment variables

The container sets these by default:

- `NPCSH_BENCHMARK_DIR=/data/npcsh`
- `NPCSH_NPC_TEAM_DIR=/root/.npcsh/npc_team`
- `NPCSH_INITIALIZED=1`
- `NPCSH_ACCEPT_PERMISSIONS=1`
- `OLLAMA_HOST=http://host.docker.internal:11434`

Override them in `docker-compose.benchmark.yml` or pass `-e KEY=value` via the `run` command.
