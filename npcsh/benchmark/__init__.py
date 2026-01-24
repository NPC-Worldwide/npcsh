"""
npcsh benchmark integration for Terminal-Bench.

This module provides integration with Terminal-Bench (tbench.ai) for benchmarking
npcsh against standardized terminal/CLI agent evaluation tasks.

Usage:
    # Install terminal-bench
    pip install terminal-bench harbor

    # Run benchmarks with npcsh
    harbor run -d terminal-bench@2.0 --agent-import-path npcsh.benchmark:NpcshAgent -m anthropic/claude-sonnet-4-20250514

    # Or use the convenience function
    from npcsh.benchmark import run_benchmark
    run_benchmark(model="claude-sonnet-4-20250514", provider="anthropic")
"""

from .runner import run_benchmark, BenchmarkRunner

__all__ = ["run_benchmark", "BenchmarkRunner"]

# NpcshAgent requires harbor to be installed - import lazily
try:
    from .npcsh_agent import NpcshAgent
    __all__.append("NpcshAgent")
except ImportError:
    NpcshAgent = None  # Harbor not installed
