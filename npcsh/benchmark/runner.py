"""
Benchmark runner for npcsh on Terminal-Bench.

Provides a convenient interface for running Terminal-Bench evaluations
with different models and providers.
"""

import subprocess
import sys
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any



@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""
    model: str = "claude-sonnet-4-20250514"
    provider: str = "anthropic"
    dataset: str = "terminal-bench"
    dataset_version: Optional[str] = None  # If None, use latest
    n_concurrent: int = 4
    task_ids: Optional[List[str]] = None
    output_dir: Optional[str] = None
    npc_name: Optional[str] = None  # Use specific NPC
    timeout: int = 600  # Per-task timeout in seconds
    extra_args: List[str] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    success: bool
    total_tasks: int = 0
    passed_tasks: int = 0
    failed_tasks: int = 0
    accuracy: float = 0.0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    duration_seconds: float = 0.0
    output_dir: str = ""
    error: Optional[str] = None
    task_results: List[Dict[str, Any]] = field(default_factory=list)


class BenchmarkRunner:
    """
    Runner for Terminal-Bench evaluations with npcsh.

    Example usage:
        runner = BenchmarkRunner()

        # Run with default settings (Claude Sonnet)
        result = runner.run()

        # Run with specific model
        result = runner.run(model="gpt-4o", provider="openai")

        # Compare multiple models
        results = runner.compare_models([
            ("claude-sonnet-4-20250514", "anthropic"),
            ("gpt-4o", "openai"),
            ("gemini-2.0-flash", "gemini"),
        ])
    """

    def __init__(self, output_base_dir: Optional[str] = None):
        """
        Initialize the benchmark runner.

        Args:
            output_base_dir: Base directory for benchmark outputs.
                           Defaults to ~/.npcsh/benchmarks/
        """
        if output_base_dir:
            self.output_base_dir = Path(output_base_dir)
        else:
            self.output_base_dir = Path.home() / ".npcsh" / "benchmarks"

        self.output_base_dir.mkdir(parents=True, exist_ok=True)

    def check_dependencies(self) -> Dict[str, bool]:
        """Check if required dependencies are installed."""
        import shutil

        deps = {
            "harbor": False,
            "terminal-bench": False,
            "docker": False,
        }

        # Find binaries in the same Python environment as current interpreter
        # Use sys.prefix to get the virtualenv/pyenv directory (don't resolve symlinks)
        bin_dir = Path(sys.prefix) / "bin"
        if not bin_dir.exists():
            # Fallback: use executable's directory without resolving
            bin_dir = Path(sys.executable).parent

        # Check harbor - first in current Python's bin dir, then PATH
        harbor_bin = bin_dir / "harbor"
        if not harbor_bin.exists():
            harbor_bin = shutil.which("harbor")

        if harbor_bin:
            try:
                result = subprocess.run(
                    [str(harbor_bin), "--version"],
                    capture_output=True,
                    text=True
                )
                deps["harbor"] = result.returncode == 0
            except (FileNotFoundError, OSError):
                pass

        # Check terminal-bench (tb CLI) - first in current Python's bin dir, then PATH
        tb_bin = bin_dir / "tb"
        if not tb_bin.exists():
            tb_bin = shutil.which("tb")

        if tb_bin:
            try:
                result = subprocess.run(
                    [str(tb_bin), "--help"],
                    capture_output=True,
                    text=True
                )
                deps["terminal-bench"] = result.returncode == 0
            except (FileNotFoundError, OSError):
                pass

        # Check docker
        try:
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                text=True
            )
            deps["docker"] = result.returncode == 0
        except FileNotFoundError:
            pass

        return deps

    def install_dependencies(self) -> bool:
        """Install Terminal-Bench dependencies."""
        print("Installing Terminal-Bench dependencies...")

        try:
            # Install harbor and terminal-bench via pip/uv
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "harbor", "terminal-bench"],
                check=True
            )
            print("Dependencies installed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to install dependencies: {e}")
            return False

    def run(
        self,
        model: str = "claude-sonnet-4-20250514",
        provider: str = "anthropic",
        dataset: str = "terminal-bench",
        dataset_version: Optional[str] = None,
        n_concurrent: int = 4,
        task_ids: Optional[List[str]] = None,
        n_tasks: Optional[int] = None,
        npc_name: Optional[str] = None,
        timeout: int = 600,
    ) -> BenchmarkResult:
        """
        Run Terminal-Bench evaluation with npcsh.

        Args:
            model: Model name (e.g., "claude-sonnet-4-20250514", "gpt-4o")
            provider: Provider name (e.g., "anthropic", "openai", "gemini")
            dataset: Dataset name (default: "terminal-bench")
            dataset_version: Dataset version (optional, uses latest if None)
            n_concurrent: Number of concurrent task executions
            task_ids: Optional list of specific task IDs to run
            n_tasks: Optional limit on number of tasks to run
            npc_name: Optional NPC name to use (e.g., "sibiji", "corca")
            timeout: Per-task timeout in seconds

        Returns:
            BenchmarkResult with evaluation metrics
        """
        # Check dependencies
        deps = self.check_dependencies()
        if not deps["harbor"]:
            print("Harbor not installed. Installing...")
            if not self.install_dependencies():
                return BenchmarkResult(
                    success=False,
                    error="Failed to install dependencies"
                )

        # Create output directory for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{provider}_{model}_{timestamp}".replace("/", "_")
        output_dir = self.output_base_dir / run_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build the harbor command
        full_model = f"{provider}/{model}"

        # Choose agent based on whether NPC is specified
        if npc_name:
            agent_path = "npcsh.benchmark:NpcshAgentWithNpc"
        else:
            agent_path = "npcsh.benchmark:NpcshAgent"

        # Find harbor in the same Python environment as current interpreter
        # Use sys.prefix to get the virtualenv/pyenv directory (don't resolve symlinks)
        import shutil
        bin_dir = Path(sys.prefix) / "bin"
        if not bin_dir.exists():
            bin_dir = Path(sys.executable).parent
        harbor_bin = str(bin_dir / "harbor")
        if not Path(harbor_bin).exists():
            harbor_bin = shutil.which("harbor") or "harbor"

        # Build dataset string (with optional version)
        dataset_str = f"{dataset}@{dataset_version}" if dataset_version else dataset

        cmd = [
            harbor_bin, "run",
            "-d", dataset_str,
            "--agent-import-path", agent_path,
            "-m", full_model,
            "-n", str(n_concurrent),
            "-o", str(output_dir),
        ]

        if task_ids:
            for task_id in task_ids:
                cmd.extend(["--task-name", task_id])

        if n_tasks:
            cmd.extend(["-l", str(n_tasks)])

        print("\nRunning Terminal-Bench evaluation:")
        print(f"  Model: {full_model}")
        print(f"  Dataset: {dataset_str}")
        print(f"  Concurrent tasks: {n_concurrent}")
        if n_tasks:
            print(f"  Max tasks: {n_tasks}")
        print(f"  Output: {output_dir}")
        if npc_name:
            print(f"  NPC: {npc_name}")
        print(f"\nCommand: {' '.join(cmd)}\n")

        start_time = datetime.now()

        try:
            # Run the benchmark
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout * n_concurrent * 2  # Overall timeout
            )

            duration = (datetime.now() - start_time).total_seconds()

            # Parse results
            result = self._parse_results(output_dir, duration)
            result.output_dir = str(output_dir)

            if process.returncode != 0:
                result.error = process.stderr

            # Save run metadata
            self._save_run_metadata(output_dir, {
                "model": full_model,
                "provider": provider,
                "dataset": dataset,
                "dataset_version": dataset_version,
                "n_concurrent": n_concurrent,
                "npc_name": npc_name,
                "duration_seconds": duration,
                "result": {
                    "success": result.success,
                    "accuracy": result.accuracy,
                    "total_tasks": result.total_tasks,
                    "passed_tasks": result.passed_tasks,
                }
            })

            return result

        except subprocess.TimeoutExpired:
            return BenchmarkResult(
                success=False,
                error="Benchmark timed out",
                output_dir=str(output_dir)
            )
        except Exception as e:
            return BenchmarkResult(
                success=False,
                error=str(e),
                output_dir=str(output_dir)
            )

    def _parse_results(self, output_dir: Path, duration: float) -> BenchmarkResult:
        """Parse benchmark results from output directory."""
        result = BenchmarkResult(
            success=True,
            duration_seconds=duration
        )

        # Look for results file
        results_file = output_dir / "results.json"
        if results_file.exists():
            try:
                with open(results_file) as f:
                    data = json.load(f)

                result.total_tasks = data.get("total", 0)
                result.passed_tasks = data.get("passed", 0)
                result.failed_tasks = data.get("failed", 0)

                if result.total_tasks > 0:
                    result.accuracy = result.passed_tasks / result.total_tasks

                result.task_results = data.get("tasks", [])

                # Aggregate token usage
                for task in result.task_results:
                    result.total_tokens += task.get("tokens", 0)
                    result.total_cost_usd += task.get("cost_usd", 0.0)

            except (json.JSONDecodeError, KeyError) as e:
                result.error = f"Failed to parse results: {e}"

        return result

    def _save_run_metadata(self, output_dir: Path, metadata: Dict[str, Any]) -> None:
        """Save run metadata to output directory."""
        metadata_file = output_dir / "run_metadata.json"
        metadata["timestamp"] = datetime.now().isoformat()

        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

    def compare_models(
        self,
        models: List[tuple],
        dataset: str = "terminal-bench",
        dataset_version: Optional[str] = None,
        n_concurrent: int = 4,
        task_ids: Optional[List[str]] = None,
    ) -> Dict[str, BenchmarkResult]:
        """
        Compare multiple models on the same benchmark.

        Args:
            models: List of (model, provider) tuples
            dataset: Dataset name
            dataset_version: Dataset version (optional)
            n_concurrent: Number of concurrent tasks
            task_ids: Optional specific task IDs

        Returns:
            Dictionary mapping model names to results

        Example:
            results = runner.compare_models([
                ("claude-sonnet-4-20250514", "anthropic"),
                ("gpt-4o", "openai"),
                ("gemini-2.0-flash", "gemini"),
            ])
        """
        results = {}

        for model, provider in models:
            print("\n" + '='*60)
            print(f"Evaluating: {provider}/{model}")
            print('='*60)

            result = self.run(
                model=model,
                provider=provider,
                dataset=dataset,
                dataset_version=dataset_version,
                n_concurrent=n_concurrent,
                task_ids=task_ids,
            )

            results[f"{provider}/{model}"] = result

            print(f"\nResult for {provider}/{model}:")
            print(f"  Accuracy: {result.accuracy:.1%}")
            print(f"  Tasks: {result.passed_tasks}/{result.total_tasks}")
            print(f"  Duration: {result.duration_seconds:.1f}s")

        # Print comparison summary
        self._print_comparison_summary(results)

        return results

    def _print_comparison_summary(self, results: Dict[str, BenchmarkResult]) -> None:
        """Print a comparison summary table."""
        print("\n" + '='*60)
        print("COMPARISON SUMMARY")
        print('='*60)
        print(f"{'Model':<40} {'Accuracy':>10} {'Tasks':>10}")
        print("-" * 60)

        sorted_results = sorted(
            results.items(),
            key=lambda x: x[1].accuracy,
            reverse=True
        )

        for model_name, result in sorted_results:
            print(
                f"{model_name:<40} "
                f"{result.accuracy:>9.1%} "
                f"{result.passed_tasks:>4}/{result.total_tasks:<4}"
            )

    def list_past_runs(self) -> List[Dict[str, Any]]:
        """List all past benchmark runs."""
        runs = []

        for run_dir in self.output_base_dir.iterdir():
            if run_dir.is_dir():
                metadata_file = run_dir / "run_metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file) as f:
                            metadata = json.load(f)
                            metadata["run_dir"] = str(run_dir)
                            runs.append(metadata)
                    except json.JSONDecodeError:
                        pass

        return sorted(runs, key=lambda x: x.get("timestamp", ""), reverse=True)


def run_benchmark(
    model: str = "claude-sonnet-4-20250514",
    provider: str = "anthropic",
    **kwargs
) -> BenchmarkResult:
    """
    Convenience function to run a Terminal-Bench evaluation.

    Args:
        model: Model name
        provider: Provider name
        **kwargs: Additional arguments passed to BenchmarkRunner.run()

    Returns:
        BenchmarkResult

    Example:
        from npcsh.benchmark import run_benchmark

        # Run with Claude
        result = run_benchmark("claude-sonnet-4-20250514", "anthropic")
        print(f"Accuracy: {result.accuracy:.1%}")

        # Run with GPT-4
        result = run_benchmark("gpt-4o", "openai")
    """
    runner = BenchmarkRunner()
    return runner.run(model=model, provider=provider, **kwargs)


def quick_test(
    model: str = "claude-sonnet-4-20250514",
    provider: str = "anthropic",
    n_tasks: int = 3,
) -> BenchmarkResult:
    """
    Run a quick test with a few tasks to verify setup.

    This runs only a few tasks to quickly verify that everything is working.
    """
    runner = BenchmarkRunner()

    # Use -l flag to limit number of tasks instead of specifying task names
    # This avoids issues with task names changing in the dataset
    return runner.run(
        model=model,
        provider=provider,
        n_concurrent=1,
        n_tasks=n_tasks,
    )


def main():
    """CLI entry point for npcsh-bench command."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Terminal-Bench with npcsh",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check dependencies
  npcsh-bench --check

  # Quick test with Claude
  npcsh-bench --quick -m claude-sonnet-4-20250514 -p anthropic

  # Full benchmark run
  npcsh-bench -m gpt-4o -p openai -n 8

  # List past runs
  npcsh-bench --list-runs

  # Compare models (requires manual setup)
  npcsh-bench --compare
        """
    )
    parser.add_argument("--model", "-m",
                       help="Model name")
    parser.add_argument("--provider", "-p",
                       help="Provider name")
    parser.add_argument("--dataset", "-d", default="terminal-bench",
                       help="Dataset name")
    parser.add_argument("--version", "-v", default=None,
                       help="Dataset version (optional, uses latest if not specified)")
    parser.add_argument("--concurrent", "-n", type=int, default=4,
                       help="Number of concurrent tasks")
    parser.add_argument("--npc", help="NPC name to use")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick test with few tasks")
    parser.add_argument("--list-runs", action="store_true",
                       help="List past benchmark runs")
    parser.add_argument("--check", action="store_true",
                       help="Check if dependencies are installed")
    parser.add_argument("--compare", action="store_true",
                       help="Compare multiple models (Claude, GPT-4, Gemini)")

    args = parser.parse_args()

    runner = BenchmarkRunner()

    if args.check:
        print("Checking Terminal-Bench dependencies...\n")
        deps = runner.check_dependencies()
        all_good = True
        for dep, installed in deps.items():
            status = "OK" if installed else "MISSING"
            symbol = "+" if installed else "-"
            print(f"  [{symbol}] {dep}: {status}")
            if not installed:
                all_good = False

        if not all_good:
            print("\nTo install missing dependencies:")
            print("  pip install harbor terminal-bench")
            print("\nOr install with npcsh:")
            print("  pip install npcsh[bench]")
        else:
            print("\nAll dependencies installed!")

    elif args.list_runs:
        runs = runner.list_past_runs()
        if not runs:
            print("No past benchmark runs found.")
        else:
            print(f"Found {len(runs)} past runs:\n")
            for run in runs:
                print(f"  {run.get('timestamp', 'unknown')}: {run.get('model', 'unknown')}")
                result = run.get('result', {})
                print(f"    Accuracy: {result.get('accuracy', 0):.1%}")
                print(f"    Tasks: {result.get('passed_tasks', 0)}/{result.get('total_tasks', 0)}")
                print()

    elif args.compare:
        print("Comparing models on Terminal-Bench 2.0...\n")
        models_to_compare = [

            ("gpt-4o", "openai"),
            ("gemini-2.0-flash", "gemini"),
        ]
        runner.compare_models(
            models_to_compare,
            n_concurrent=args.concurrent
        )

    elif args.quick:
        result = quick_test(args.model, args.provider)
        print(f"\nQuick test result: {'PASS' if result.success else 'FAIL'}")
        print(f"Accuracy: {result.accuracy:.1%}")

    else:
        result = runner.run(
            model=args.model,
            provider=args.provider,
            dataset=args.dataset,
            dataset_version=args.version,
            n_concurrent=args.concurrent,
            npc_name=args.npc,
        )
        print("\nBenchmark complete!")
        print(f"Accuracy: {result.accuracy:.1%}")
        print(f"Results saved to: {result.output_dir}")


if __name__ == "__main__":
    main()
