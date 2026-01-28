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

    def __init__(self, logs_dir: Path = None, model_name: str = None, logger=None, **kwargs):
        super().__init__(logs_dir=logs_dir, model_name=model_name, logger=logger, **kwargs)

    @staticmethod
    def name() -> str:
        return "npcsh"

    @property
    def _install_agent_template_path(self) -> Path:
        """Path to the jinja template script for installing npcsh in the container."""
        return Path(__file__).parent / "templates" / "install-npcsh.sh.j2"

    def create_run_agent_commands(self, instruction: str) -> list:
        """
        Create the commands to run npcsh in the container.

        Args:
            instruction: The task instruction from Terminal-Bench

        Returns:
            List of ExecInput commands to execute
        """
        # Wrap the instruction with explicit jinx usage directions and retry logic
        tool_instruction = f"""You have access to jinxs including edit_file (for writing/creating files), sh (for running shell commands), and python (for running Python code).

IMPORTANT RULES:
1. You MUST use these jinxs to complete the task. Do NOT just output code as text - use the edit_file jinx to actually write files to disk.
2. After implementing a solution, you MUST verify it works by running any provided test scripts.
3. If a test fails or produces an error, you MUST try a DIFFERENT approach. Do not give up.
4. Keep trying different approaches until you succeed or have tried at least 10 different solutions.
5. NEVER assume success - always check the actual output of test commands.

Task: {instruction}

WORKFLOW:
1. Implement your solution using edit_file and sh
2. Run any test scripts mentioned in the task
3. Check the output carefully - look for "PASS", "SUCCESS", "OK" or similar
4. If the test failed, analyze why and try a completely different approach
5. Repeat until the test passes

Remember: Use edit_file to write code files. Use sh to run commands. VERIFY your solution works before concluding."""

        escaped_instruction = shlex.quote(tool_instruction)
        model_name = self.model_name

        if model_name and "/" in model_name:
            provider, model = model_name.split("/", 1)
        elif model_name:
            provider = os.environ.get("NPCSH_CHAT_PROVIDER", "")
            model = model_name
        else:
            provider = os.environ.get("NPCSH_CHAT_PROVIDER", "")
            model = os.environ.get("NPCSH_CHAT_MODEL", "")

        # Map provider names to npcsh provider format
        provider_map = {
            "anthropic": "anthropic",
            "openai": "openai",
            "google": "gemini",
            "gemini": "gemini",
            "deepseek": "deepseek",
            "ollama": "ollama",
            "groq": "groq",
            "openrouter": "openrouter",
        }
        npcsh_provider = provider_map.get(provider.lower(), provider)

        # Build environment variables for API keys
        env_vars = []
        api_key_map = {
            "anthropic": ["ANTHROPIC_API_KEY"],
            "openai": ["OPENAI_API_KEY"],
            "gemini": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
            "google": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
            "deepseek": ["DEEPSEEK_API_KEY"],
            "groq": ["GROQ_API_KEY"],
            "openrouter": ["OPENROUTER_API_KEY"],
        }

        added_keys = set()
        for prov, env_keys in api_key_map.items():
            for env_key in env_keys:
                if env_key in os.environ:
                    # For Gemini, always pass as GOOGLE_API_KEY (what litellm expects)
                    target_key = "GOOGLE_API_KEY" if env_key == "GEMINI_API_KEY" else env_key
                    if target_key not in added_keys:
                        env_vars.append(f'{target_key}="{os.environ[env_key]}"')
                        added_keys.add(target_key)
                    break

        env_prefix = " ".join(env_vars) + " " if env_vars else ""

        # Output directory for logs
        output_dir = str(self.logs_dir / "npcsh_output")
        output_file = str(self.logs_dir / "npcsh_output" / "output.jsonl")

        commands = []

        # Create output directory
        commands.append(ExecInput(
            command=f"mkdir -p {shlex.quote(output_dir)}",
            timeout_sec=30
        ))

        # Create .npcsh_global file to use global team and avoid interactive prompts
        commands.append(ExecInput(
            command="touch /app/.npcsh_global",
            timeout_sec=10
        ))

        # Run npcsh with the instruction
        # Using corca NPC which has edit_file tool for writing files
        # Using the npc CLI which supports single-command execution
        # NPCSH_DEFAULT_MODE=agent enables automatic tool execution
        npcsh_cmd = (
            f'{env_prefix}'
            f'NPCSH_CHAT_MODEL="{model}" '
            f'NPCSH_CHAT_PROVIDER="{npcsh_provider}" '
            f'NPCSH_STREAM_OUTPUT=0 '
            f'NPCSH_DEFAULT_MODE=agent '
            f'npc --npc corca {escaped_instruction} '
            f'2>&1 | tee {shlex.quote(output_file)}'
        )

        commands.append(ExecInput(
            command=npcsh_cmd,
            timeout_sec=600,  # 10 minute timeout for complex tasks
        ))

        return commands

    def populate_context_post_run(self, context: AgentContext) -> None:
        """
        Populate the context with results of the agent execution.

        Parses the output file to extract token usage metrics.

        Args:
            context: The AgentContext to populate with metrics
        """
        output_file = self.logs_dir / "npcsh_output" / "output.jsonl"

        total_input_tokens = 0
        total_output_tokens = 0
        total_cost_usd = 0.0

        if output_file.exists():
            try:
                with open(output_file, 'r') as f:
                    content = f.read()

                # Try to parse as JSONL first
                for line in content.strip().split('\n'):
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                        # Extract token usage from events if present
                        if isinstance(event, dict):
                            usage = event.get('usage', {})
                            total_input_tokens += usage.get('input_tokens', 0)
                            total_output_tokens += usage.get('output_tokens', 0)
                            total_cost_usd += usage.get('cost_usd', 0.0)
                    except json.JSONDecodeError:
                        # Not JSON, just regular output
                        pass

            except Exception as e:
                self.logger.warning(f"Failed to parse npcsh output: {e}")

        # Set context metrics
        if hasattr(context, 'input_tokens'):
            context.input_tokens = total_input_tokens
        if hasattr(context, 'output_tokens'):
            context.output_tokens = total_output_tokens
        if hasattr(context, 'cost_usd'):
            context.cost_usd = total_cost_usd


class NpcshAgentWithNpc(NpcshAgent):
    """
    Variant that uses a specific NPC for task execution.

    This allows benchmarking specific NPCs like sibiji (orchestrator),
    corca (coding), or custom NPCs.

    Usage:
        harbor run -d terminal-bench@2.0 \\
            --agent-import-path "npcsh.benchmark:NpcshAgentWithNpc" \\
            -m anthropic/claude-sonnet-4-20250514 -n 4
    """

    def __init__(self, *args, npc_name: str = "sibiji", **kwargs):
        super().__init__(*args, **kwargs)
        self.npc_name = npc_name

    @staticmethod
    def name() -> str:
        return "npcsh-npc"

    def create_run_agent_commands(self, instruction: str) -> list:
        """Create commands using a specific NPC."""
        # Wrap the instruction with explicit jinx usage directions and retry logic
        tool_instruction = f"""You have access to jinxs including edit_file (for writing/creating files), sh (for running shell commands), and python (for running Python code).

IMPORTANT RULES:
1. You MUST use these jinxs to complete the task. Do NOT just output code as text - use the edit_file jinx to actually write files to disk.
2. After implementing a solution, you MUST verify it works by running any provided test scripts.
3. If a test fails or produces an error, you MUST try a DIFFERENT approach. Do not give up.
4. Keep trying different approaches until you succeed or have tried at least 10 different solutions.
5. NEVER assume success - always check the actual output of test commands.

Task: {instruction}

WORKFLOW:
1. Implement your solution using edit_file and sh
2. Run any test scripts mentioned in the task
3. Check the output carefully - look for "PASS", "SUCCESS", "OK" or similar
4. If the test failed, analyze why and try a completely different approach
5. Repeat until the test passes

Remember: Use edit_file to write code files. Use sh to run commands. VERIFY your solution works before concluding."""

        escaped_instruction = shlex.quote(tool_instruction)
        model_name = self.model_name

        if model_name and "/" in model_name:
            provider, model = model_name.split("/", 1)
        elif model_name:
            provider = os.environ.get("NPCSH_CHAT_PROVIDER", "")
            model = model_name
        else:
            provider = os.environ.get("NPCSH_CHAT_PROVIDER", "")
            model = os.environ.get("NPCSH_CHAT_MODEL", "")

        provider_map = {
            "anthropic": "anthropic",
            "openai": "openai",
            "google": "gemini",
            "gemini": "gemini",
            "deepseek": "deepseek",
            "ollama": "ollama",
        }
        npcsh_provider = provider_map.get(provider.lower(), provider)

        env_vars = []
        api_key_map = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "gemini": "GOOGLE_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
        }

        for prov, env_key in api_key_map.items():
            if env_key in os.environ:
                env_vars.append(f'{env_key}="{os.environ[env_key]}"')

        env_prefix = " ".join(env_vars) + " " if env_vars else ""

        output_dir = str(self.logs_dir / "npcsh_output")
        output_file = str(self.logs_dir / "npcsh_output" / "output.jsonl")

        commands = []

        commands.append(ExecInput(
            command=f"mkdir -p {shlex.quote(output_dir)}",
            timeout_sec=30
        ))

        # Create .npcsh_global file to use global team and avoid interactive prompts
        commands.append(ExecInput(
            command="touch /app/.npcsh_global",
            timeout_sec=10
        ))

        # Use specific NPC with --npc flag
        # NPCSH_DEFAULT_MODE=agent enables automatic tool execution
        npcsh_cmd = (
            f'{env_prefix}'
            f'NPCSH_CHAT_MODEL="{model}" '
            f'NPCSH_CHAT_PROVIDER="{npcsh_provider}" '
            f'NPCSH_STREAM_OUTPUT=0 '
            f'NPCSH_DEFAULT_MODE=agent '
            f'npc --npc {self.npc_name} {escaped_instruction} '
            f'2>&1 | tee {shlex.quote(output_file)}'
        )

        commands.append(ExecInput(
            command=npcsh_cmd,
            timeout_sec=600,
        ))

        return commands
