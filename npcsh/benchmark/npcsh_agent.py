"""
npcsh Harbor Agent Adapter for Terminal-Bench.

This module implements the BaseInstalledAgent interface for running npcsh
as an agent in Terminal-Bench evaluations.
"""

import json
import os
import shlex
from pathlib import Path

from harbor.agents.installed.base import BaseInstalledAgent, ExecInput
from harbor.models.agent.context import AgentContext


class NpcshAgent(BaseInstalledAgent):
    """
    Harbor agent adapter for npcsh.

    This allows npcsh to be evaluated on Terminal-Bench tasks by:
    1. Installing npcsh in the benchmark container
    2. Running npcsh with the task instruction
    3. Parsing output for token usage and results

    Usage:
        harbor run -d terminal-bench@2.0 \\
            --agent-import-path npcsh.benchmark:NpcshAgent \\
            -m anthropic/claude-sonnet-4-20250514 -n 4
    """

    SUPPORTS_ATIF = True  # Agent Trajectory Interchange Format

    # Root of the npcsh source tree (two levels up from this file)
    _NPCSH_SRC = Path(__file__).resolve().parent.parent.parent

    def __init__(self, logs_dir: Path = None, model_name: str = None, logger=None, **kwargs):
        super().__init__(logs_dir=logs_dir, model_name=model_name, logger=logger, **kwargs)

    @staticmethod
    def name() -> str:
        return "npcsh"

    @property
    def _install_agent_template_path(self) -> Path:
        """Path to the jinja template script for installing npcsh in the container."""
        return Path(__file__).parent / "templates" / "install-npcsh.sh.j2"

    async def setup(self, environment) -> None:
        """Upload local npcsh + npcpy source, then run install script."""
        import shutil
        import tempfile

        npcsh_src = self._NPCSH_SRC
        npcpy_src = npcsh_src.parent / "npcpy"

        # Create /src in container
        await environment.exec(command="mkdir -p /src")

        # Copy source to temp dir excluding .git and caches
        def _copy_clean(src, name):
            tmp = Path(tempfile.mkdtemp()) / name
            shutil.copytree(
                src, tmp,
                ignore=shutil.ignore_patterns(
                    '.git', '__pycache__', '*.pyc', 'dist', 'build',
                    '*.egg-info', 'jobs', 'dataset_cache',
                ),
            )
            return tmp

        clean_npcsh = _copy_clean(npcsh_src, "npcsh")
        await environment.upload_dir(
            source_dir=str(clean_npcsh),
            target_dir="/src/npcsh",
        )

        if npcpy_src.exists():
            clean_npcpy = _copy_clean(npcpy_src, "npcpy")
            await environment.upload_dir(
                source_dir=str(clean_npcpy),
                target_dir="/src/npcpy",
            )

        await super().setup(environment)

    def create_run_agent_commands(self, instruction: str) -> list:
        """Run instruction through npcsh -c, which handles everything."""
        escaped_instruction = shlex.quote(instruction)

        # Forward env vars into the container — npcsh reads these directly
        env_vars = []
        for key in [
            "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY",
            "GEMINI_API_KEY", "DEEPSEEK_API_KEY", "GROQ_API_KEY",
            "OPENROUTER_API_KEY", "OLLAMA_HOST",
        ]:
            val = os.environ.get(key)
            if val:
                env_vars.append(f'{key}="{val}"')

        # Model/provider from Harbor's model_name (e.g. "ollama/phi4")
        model_name = self.model_name or ""
        if "/" in model_name:
            provider, model = model_name.split("/", 1)
        else:
            provider = os.environ.get("NPCSH_CHAT_PROVIDER", "")
            model = model_name or os.environ.get("NPCSH_CHAT_MODEL", "")

        env_vars.append(f'NPCSH_CHAT_MODEL="{model}"')
        env_vars.append(f'NPCSH_CHAT_PROVIDER="{provider}"')
        env_vars.append('NPCSH_STREAM_OUTPUT=0')
        env_vars.append('NPCSH_DEBUG=1')

        if provider == "ollama":
            if "OLLAMA_HOST" not in os.environ:
                env_vars.append('OLLAMA_HOST="http://host.docker.internal:11434"')
            # Use 16k context for ollama models to avoid losing instructions
            env_vars.append('NPCSH_OLLAMA_NUM_CTX=16384')

        env_prefix = " ".join(env_vars)
        output_dir = str(self.logs_dir / "npcsh_output")
        output_file = str(self.logs_dir / "npcsh_output" / "output.jsonl")

        return [
            ExecInput(command=f"mkdir -p {shlex.quote(output_dir)}", timeout_sec=30),
            ExecInput(command="touch /app/.npcsh_global", timeout_sec=10),
            ExecInput(
                command=f'{env_prefix} npcsh -c {escaped_instruction} 2>&1 | tee {shlex.quote(output_file)}',
                timeout_sec=1800,
            ),
        ]

    def populate_context_post_run(self, context: AgentContext) -> None:
        """
        Populate the context with results of the agent execution.

        Parses the output file to extract token usage metrics.
        """
        output_file = self.logs_dir / "npcsh_output" / "output.jsonl"

        total_input_tokens = 0
        total_output_tokens = 0
        total_cost_usd = 0.0

        if output_file.exists():
            try:
                with open(output_file, 'r') as f:
                    content = f.read()

                for line in content.strip().split('\n'):
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                        if isinstance(event, dict):
                            usage = event.get('usage', {})
                            total_input_tokens += usage.get('input_tokens', 0)
                            total_output_tokens += usage.get('output_tokens', 0)
                            total_cost_usd += usage.get('cost_usd', 0.0)
                    except json.JSONDecodeError:
                        pass

            except Exception as e:
                self.logger.warning(f"Failed to parse npcsh output: {e}")

        if hasattr(context, 'input_tokens'):
            context.input_tokens = total_input_tokens
        if hasattr(context, 'output_tokens'):
            context.output_tokens = total_output_tokens
        if hasattr(context, 'cost_usd'):
            context.cost_usd = total_cost_usd
