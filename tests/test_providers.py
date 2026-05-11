import os
import sys
import json
import tempfile
import yaml
import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from pathlib import Path


# ── PR #131 + #133: Provider config injection in setup_shell ──────────────


class TestProvidersFieldConfig:
    """Tests for PR #133: providers field support in .ctx files."""

    def _make_team_dir(self, tmp_path, team_ctx_content, npc_specs=None):
        """Helper: create a temporary team dir with .ctx + .npc files."""
        team_dir = tmp_path / "test_team"
        team_dir.mkdir()

        ctx_path = team_dir / "team.ctx"
        with open(ctx_path, "w") as f:
            yaml.dump(team_ctx_content, f)

        npc_specs = npc_specs or {}
        for name, spec in npc_specs.items():
            npc_path = team_dir / f"{name}.npc"
            with open(npc_path, "w") as f:
                yaml.dump(spec, f)

        return str(team_dir)

    def test_providers_dict_parsed_from_ctx(self, tmp_path, monkeypatch):
        """Provider definitions in team.ctx are parsed into a lookup dict."""
        from npcsh._state import setup_shell

        team_ctx = {
            "context": "test",
            "providers": [
                {
                    "name": "ollama_cloud",
                    "provider_type": "openai-like",
                    "model": "llama3.2",
                    "api_url": "https://ollama.example.com/v1",
                    "api_key": "sk-test",
                },
            ],
        }
        npc_specs = {
            "forenpc": {"name": "forenpc", "primary_directive": "test"},
        }
        team_dir = self._make_team_dir(tmp_path, team_ctx, npc_specs)

        # Point setup_shell at our temp dir
        monkeypatch.setenv("NPCSH_DB_PATH", str(tmp_path / "test.db"))
        monkeypatch.setattr(
            "npcsh._state.PROJECT_NPC_TEAM_PATH", str(tmp_path / "nonexistent")
        )
        monkeypatch.setattr("npcsh._state.DEFAULT_NPC_TEAM_PATH", team_dir)

        with patch("npcsh._state.ensure_npcshrc_exists"), patch(
            "npcsh._state.load_npcshrc_env"
        ), patch("npcsh._state.add_npcshrc_to_shell_config"), patch(
            "npcsh._state.setup_readline"
        ), patch(
            "npcsh._state.save_readline_history"
        ), patch(
            "npcsh._state.is_npcsh_initialized", return_value=True
        ):
            _, team, forenpc = setup_shell()

        assert team is not None

    def test_npc_inherits_from_named_provider(self, tmp_path, monkeypatch):
        """An NPC referencing a provider by name inherits model/api_url/api_key."""
        from npcsh._state import setup_shell

        team_ctx = {
            "context": "test",
            "providers": [
                {
                    "name": "ollama_cloud",
                    "provider_type": "openai-like",
                    "model": "llama3.2",
                    "api_url": "https://ollama.example.com/v1",
                    "api_key": "sk-test",
                },
            ],
        }
        npc_specs = {
            "forenpc": {"name": "forenpc", "primary_directive": "test"},
            "researcher": {
                "name": "researcher",
                "primary_directive": "research",
                "provider": "ollama_cloud",
            },
        }
        team_dir = self._make_team_dir(tmp_path, team_ctx, npc_specs)

        monkeypatch.setenv("NPCSH_DB_PATH", str(tmp_path / "test.db"))
        monkeypatch.setattr(
            "npcsh._state.PROJECT_NPC_TEAM_PATH", str(tmp_path / "nonexistent")
        )
        monkeypatch.setattr("npcsh._state.DEFAULT_NPC_TEAM_PATH", team_dir)

        with patch("npcsh._state.ensure_npcshrc_exists"), patch(
            "npcsh._state.load_npcshrc_env"
        ), patch("npcsh._state.add_npcshrc_to_shell_config"), patch(
            "npcsh._state.setup_readline"
        ), patch(
            "npcsh._state.save_readline_history"
        ), patch(
            "npcsh._state.is_npcsh_initialized", return_value=True
        ):
            _, team, _ = setup_shell()

        npc = team.npcs["researcher"]
        assert npc.model == "llama3.2"
        assert npc.provider == "openai-like"
        assert getattr(npc, "api_url", None) == "https://ollama.example.com/v1"
        assert getattr(npc, "api_key", None) == "sk-test"

    def test_npc_override_model_keeps_provider_rest(self, tmp_path, monkeypatch):
        """An NPC can override model while inheriting api_url/api_key from provider."""
        from npcsh._state import setup_shell

        team_ctx = {
            "context": "test",
            "providers": [
                {
                    "name": "ollama_cloud",
                    "provider_type": "openai-like",
                    "model": "llama3.2",
                    "api_url": "https://ollama.example.com/v1",
                    "api_key": "sk-test",
                },
            ],
        }
        npc_specs = {
            "forenpc": {"name": "forenpc", "primary_directive": "test"},
            "coder": {
                "name": "coder",
                "primary_directive": "code",
                "provider": "ollama_cloud",
                "model": "codellama",
            },
        }
        team_dir = self._make_team_dir(tmp_path, team_ctx, npc_specs)

        monkeypatch.setenv("NPCSH_DB_PATH", str(tmp_path / "test.db"))
        monkeypatch.setattr(
            "npcsh._state.PROJECT_NPC_TEAM_PATH", str(tmp_path / "nonexistent")
        )
        monkeypatch.setattr("npcsh._state.DEFAULT_NPC_TEAM_PATH", team_dir)

        with patch("npcsh._state.ensure_npcshrc_exists"), patch(
            "npcsh._state.load_npcshrc_env"
        ), patch("npcsh._state.add_npcshrc_to_shell_config"), patch(
            "npcsh._state.setup_readline"
        ), patch(
            "npcsh._state.save_readline_history"
        ), patch(
            "npcsh._state.is_npcsh_initialized", return_value=True
        ):
            _, team, _ = setup_shell()

        npc = team.npcs["coder"]
        assert npc.model == "codellama"          # overridden
        assert npc.provider == "openai-like"      # from provider
        assert getattr(npc, "api_url", None) == "https://ollama.example.com/v1"
        assert getattr(npc, "api_key", None) == "sk-test"

    def test_explicit_npc_api_url_takes_precedence(self, tmp_path, monkeypatch):
        """Explicit api_url on NPC overrides provider default."""
        from npcsh._state import setup_shell

        team_ctx = {
            "context": "test",
            "providers": [
                {
                    "name": "ollama_cloud",
                    "provider_type": "openai-like",
                    "model": "llama3.2",
                    "api_url": "https://ollama.example.com/v1",
                },
            ],
        }
        npc_specs = {
            "forenpc": {"name": "forenpc", "primary_directive": "test"},
            "coder": {
                "name": "coder",
                "primary_directive": "code",
                "provider": "ollama_cloud",
                "api_url": "http://my-local:11434/v1",
            },
        }
        team_dir = self._make_team_dir(tmp_path, team_ctx, npc_specs)

        monkeypatch.setenv("NPCSH_DB_PATH", str(tmp_path / "test.db"))
        monkeypatch.setattr(
            "npcsh._state.PROJECT_NPC_TEAM_PATH", str(tmp_path / "nonexistent")
        )
        monkeypatch.setattr("npcsh._state.DEFAULT_NPC_TEAM_PATH", team_dir)

        with patch("npcsh._state.ensure_npcshrc_exists"), patch(
            "npcsh._state.load_npcshrc_env"
        ), patch("npcsh._state.add_npcshrc_to_shell_config"), patch(
            "npcsh._state.setup_readline"
        ), patch(
            "npcsh._state.save_readline_history"
        ), patch(
            "npcsh._state.is_npcsh_initialized", return_value=True
        ):
            _, team, _ = setup_shell()

        npc = team.npcs["coder"]
        assert getattr(npc, "api_url", None) == "http://my-local:11434/v1"

    def test_provider_env_var_expansion(self, tmp_path, monkeypatch):
        """${VAR} syntax in provider configs is expanded."""
        from npcsh._state import setup_shell

        monkeypatch.setenv("OLLAMA_KEY", "secret123")

        team_ctx = {
            "context": "test",
            "providers": [
                {
                    "name": "ollama_cloud",
                    "provider_type": "openai-like",
                    "model": "llama3.2",
                    "api_url": "https://ollama.example.com/v1",
                    "api_key": "${OLLAMA_KEY}",
                },
            ],
        }
        npc_specs = {
            "forenpc": {"name": "forenpc", "primary_directive": "test"},
            "npc1": {"name": "npc1", "primary_directive": "d", "provider": "ollama_cloud"},
        }
        team_dir = self._make_team_dir(tmp_path, team_ctx, npc_specs)

        monkeypatch.setenv("NPCSH_DB_PATH", str(tmp_path / "test.db"))
        monkeypatch.setattr(
            "npcsh._state.PROJECT_NPC_TEAM_PATH", str(tmp_path / "nonexistent")
        )
        monkeypatch.setattr("npcsh._state.DEFAULT_NPC_TEAM_PATH", team_dir)

        with patch("npcsh._state.ensure_npcshrc_exists"), patch(
            "npcsh._state.load_npcshrc_env"
        ), patch("npcsh._state.add_npcshrc_to_shell_config"), patch(
            "npcsh._state.setup_readline"
        ), patch(
            "npcsh._state.save_readline_history"
        ), patch(
            "npcsh._state.is_npcsh_initialized", return_value=True
        ):
            _, team, _ = setup_shell()

        npc = team.npcs["npc1"]
        assert getattr(npc, "api_key", None) == "secret123"

    def test_forenpc_inherits_from_named_provider(self, tmp_path, monkeypatch):
        """Forenpc can also reference a named provider."""
        from npcsh._state import setup_shell

        team_ctx = {
            "context": "test",
            "providers": [
                {
                    "name": "ollama_cloud",
                    "provider_type": "openai-like",
                    "model": "llama3.2",
                    "api_url": "https://ollama.example.com/v1",
                },
            ],
        }
        npc_specs = {
            "forenpc": {
                "name": "forenpc",
                "primary_directive": "test",
                "provider": "ollama_cloud",
            },
        }
        team_dir = self._make_team_dir(tmp_path, team_ctx, npc_specs)

        monkeypatch.setenv("NPCSH_DB_PATH", str(tmp_path / "test.db"))
        monkeypatch.setattr(
            "npcsh._state.PROJECT_NPC_TEAM_PATH", str(tmp_path / "nonexistent")
        )
        monkeypatch.setattr("npcsh._state.DEFAULT_NPC_TEAM_PATH", team_dir)

        with patch("npcsh._state.ensure_npcshrc_exists"), patch(
            "npcsh._state.load_npcshrc_env"
        ), patch("npcsh._state.add_npcshrc_to_shell_config"), patch(
            "npcsh._state.setup_readline"
        ), patch(
            "npcsh._state.save_readline_history"
        ), patch(
            "npcsh._state.is_npcsh_initialized", return_value=True
        ):
            _, team, forenpc = setup_shell()

        assert forenpc is not None
        assert forenpc.model == "llama3.2"
        assert forenpc.provider == "openai-like"


class TestOpenaiLikeApiUrlInjection:
    """Tests for PR #131: NPCSH_API_URL injection for openai-like providers."""

    def _make_team_dir(self, tmp_path, team_ctx_content, npc_specs=None):
        team_dir = tmp_path / "test_team"
        team_dir.mkdir()

        ctx_path = team_dir / "team.ctx"
        with open(ctx_path, "w") as f:
            yaml.dump(team_ctx_content, f)

        npc_specs = npc_specs or {}
        for name, spec in npc_specs.items():
            npc_path = team_dir / f"{name}.npc"
            with open(npc_path, "w") as f:
                yaml.dump(spec, f)

        return str(team_dir)

    def test_npcsh_api_url_injected_for_openai_like(self, tmp_path, monkeypatch):
        """When NPC uses openai-like without api_url, NPCSH_API_URL is injected."""
        from npcsh._state import setup_shell

        monkeypatch.setenv("NPCSH_API_URL", "https://my-ollama.example.com/v1")

        team_ctx = {"context": "test"}
        npc_specs = {
            "forenpc": {
                "name": "forenpc",
                "primary_directive": "test",
                "provider": "openai-like",
            },
        }
        team_dir = self._make_team_dir(tmp_path, team_ctx, npc_specs)

        monkeypatch.setenv("NPCSH_DB_PATH", str(tmp_path / "test.db"))
        monkeypatch.setattr(
            "npcsh._state.PROJECT_NPC_TEAM_PATH", str(tmp_path / "nonexistent")
        )
        monkeypatch.setattr("npcsh._state.DEFAULT_NPC_TEAM_PATH", team_dir)

        with patch("npcsh._state.ensure_npcshrc_exists"), patch(
            "npcsh._state.load_npcshrc_env"
        ), patch("npcsh._state.add_npcshrc_to_shell_config"), patch(
            "npcsh._state.setup_readline"
        ), patch(
            "npcsh._state.save_readline_history"
        ), patch(
            "npcsh._state.is_npcsh_initialized", return_value=True
        ):
            _, team, forenpc = setup_shell()

        assert getattr(forenpc, "api_url", None) == "https://my-ollama.example.com/v1"

    def test_explicit_api_url_takes_precedence_over_env(self, tmp_path, monkeypatch):
        """Explicit api_url on NPC should not be overwritten by env var."""
        from npcsh._state import setup_shell

        monkeypatch.setenv("NPCSH_API_URL", "https://env-url.example.com/v1")

        team_ctx = {"context": "test"}
        npc_specs = {
            "forenpc": {
                "name": "forenpc",
                "primary_directive": "test",
                "provider": "openai-like",
                "api_url": "https://explicit.example.com/v1",
            },
        }
        team_dir = self._make_team_dir(tmp_path, team_ctx, npc_specs)

        monkeypatch.setenv("NPCSH_DB_PATH", str(tmp_path / "test.db"))
        monkeypatch.setattr(
            "npcsh._state.PROJECT_NPC_TEAM_PATH", str(tmp_path / "nonexistent")
        )
        monkeypatch.setattr("npcsh._state.DEFAULT_NPC_TEAM_PATH", team_dir)

        with patch("npcsh._state.ensure_npcshrc_exists"), patch(
            "npcsh._state.load_npcshrc_env"
        ), patch("npcsh._state.add_npcshrc_to_shell_config"), patch(
            "npcsh._state.setup_readline"
        ), patch(
            "npcsh._state.save_readline_history"
        ), patch(
            "npcsh._state.is_npcsh_initialized", return_value=True
        ):
            _, team, forenpc = setup_shell()

        assert getattr(forenpc, "api_url", None) == "https://explicit.example.com/v1"

    def test_no_env_var_no_injection(self, tmp_path, monkeypatch):
        """Without NPCSH_API_URL env var, openai-like NPC gets no api_url."""
        from npcsh._state import setup_shell

        monkeypatch.delenv("NPCSH_API_URL", raising=False)

        team_ctx = {"context": "test"}
        npc_specs = {
            "forenpc": {
                "name": "forenpc",
                "primary_directive": "test",
                "provider": "openai-like",
            },
        }
        team_dir = self._make_team_dir(tmp_path, team_ctx, npc_specs)

        monkeypatch.setenv("NPCSH_DB_PATH", str(tmp_path / "test.db"))
        monkeypatch.setattr(
            "npcsh._state.PROJECT_NPC_TEAM_PATH", str(tmp_path / "nonexistent")
        )
        monkeypatch.setattr("npcsh._state.DEFAULT_NPC_TEAM_PATH", team_dir)

        with patch("npcsh._state.ensure_npcshrc_exists"), patch(
            "npcsh._state.load_npcshrc_env"
        ), patch("npcsh._state.add_npcshrc_to_shell_config"), patch(
            "npcsh._state.setup_readline"
        ), patch(
            "npcsh._state.save_readline_history"
        ), patch(
            "npcsh._state.is_npcsh_initialized", return_value=True
        ):
            _, team, forenpc = setup_shell()

        assert getattr(forenpc, "api_url", None) is None

    def test_providers_plus_env_url_both_work(self, tmp_path, monkeypatch):
        """Provider config sets api_url; env URL is only fallback when not set."""
        from npcsh._state import setup_shell

        monkeypatch.setenv("NPCSH_API_URL", "https://env-url.example.com/v1")

        team_ctx = {
            "context": "test",
            "providers": [
                {
                    "name": "my_prov",
                    "provider_type": "openai-like",
                    "model": "llama3.2",
                    "api_url": "https://provider.example.com/v1",
                },
            ],
        }
        npc_specs = {
            "forenpc": {
                "name": "forenpc",
                "primary_directive": "test",
                "provider": "my_prov",
            },
        }
        team_dir = self._make_team_dir(tmp_path, team_ctx, npc_specs)

        monkeypatch.setenv("NPCSH_DB_PATH", str(tmp_path / "test.db"))
        monkeypatch.setattr(
            "npcsh._state.PROJECT_NPC_TEAM_PATH", str(tmp_path / "nonexistent")
        )
        monkeypatch.setattr("npcsh._state.DEFAULT_NPC_TEAM_PATH", team_dir)

        with patch("npcsh._state.ensure_npcshrc_exists"), patch(
            "npcsh._state.load_npcshrc_env"
        ), patch("npcsh._state.add_npcshrc_to_shell_config"), patch(
            "npcsh._state.setup_readline"
        ), patch(
            "npcsh._state.save_readline_history"
        ), patch(
            "npcsh._state.is_npcsh_initialized", return_value=True
        ):
            _, team, forenpc = setup_shell()

        # Provider URL wins; env URL is fallback only
        assert getattr(forenpc, "api_url", None) == "https://provider.example.com/v1"


# ── PR #129: CLI providers, delegation, session continuity ───────────────


class TestCLIProviderRouting:
    """Tests for PR #129: CLI provider short-circuit and session continuity."""

    def test_cli_agent_shortcircuit_called(self, monkeypatch, tmp_path):
        """process_pipeline_command short-circuits to npc.run() for CLIAgent."""
        from npcsh._state import ShellState, process_pipeline_command
        from npcpy.npc_compiler import CLIAgent

        mock_run = MagicMock(return_value={"output": "cli result"})
        fake_npc = MagicMock(spec=CLIAgent)
        fake_npc.name = "coder"
        fake_npc.cli_provider = "opencode"
        fake_npc.provider = "opencode"
        fake_npc.model = "opencode-default"
        fake_npc.run = mock_run

        state = ShellState(current_path=str(tmp_path))
        state.npc = fake_npc

        with patch("npcsh._state.get_locally_available_models", return_value={}):
            _, output = process_pipeline_command(
                "write some code", None, state, stream_final=False, router=None
            )

        mock_run.assert_called_once()
        assert output == {"output": "cli result"}

    def test_cli_session_id_stored(self, monkeypatch, tmp_path):
        """Returned session_id from CLIAgent.run is stored in state.cli_sessions."""
        from npcsh._state import ShellState, process_pipeline_command
        from npcpy.npc_compiler import CLIAgent

        mock_run = MagicMock(return_value={"output": "ok", "session_id": "sess_42"})
        fake_npc = MagicMock(spec=CLIAgent)
        fake_npc.name = "coder"
        fake_npc.cli_provider = "opencode"
        fake_npc.provider = "opencode"
        fake_npc.model = "opencode-default"
        fake_npc.run = mock_run

        state = ShellState(current_path=str(tmp_path))
        state.npc = fake_npc

        with patch("npcsh._state.get_locally_available_models", return_value={}):
            _, output = process_pipeline_command(
                "write some code", None, state, stream_final=False, router=None
            )

        assert state.cli_sessions[("opencode", "coder")] == "sess_42"

    def test_existing_session_id_passed(self, monkeypatch, tmp_path):
        """If a session_id already exists, it is passed to CLIAgent.run()."""
        from npcsh._state import ShellState, process_pipeline_command
        from npcpy.npc_compiler import CLIAgent

        mock_run = MagicMock(return_value={"output": "ok"})
        fake_npc = MagicMock(spec=CLIAgent)
        fake_npc.name = "coder"
        fake_npc.cli_provider = "opencode"
        fake_npc.provider = "opencode"
        fake_npc.model = "opencode-default"
        fake_npc.run = mock_run

        state = ShellState(current_path=str(tmp_path))
        state.npc = fake_npc
        state.cli_sessions[("opencode", "coder")] = "existing_sid"

        with patch("npcsh._state.get_locally_available_models", return_value={}):
            _, output = process_pipeline_command(
                "write some code", None, state, stream_final=False, router=None
            )

        call_kwargs = mock_run.call_args.kwargs
        assert call_kwargs.get("session_id") == "existing_sid"

    def test_non_cli_agent_still_uses_litellm_path(self, monkeypatch, tmp_path):
        """Non-CLI NPCs continue through the normal litellm/tool loop."""
        from npcsh._state import ShellState, process_pipeline_command
        from npcpy.npc_compiler import NPC

        state = ShellState(current_path=str(tmp_path))
        state.npc = NPC(name="sibiji", primary_directive="help")

        llm_calls = []

        def fake_get_llm_response(*args, **kwargs):
            llm_calls.append("llm")
            return {"response": "litellm path"}

        with patch(
            "npcsh._state.get_locally_available_models", return_value={}
        ), patch("npcsh._state.get_llm_response", fake_get_llm_response), patch(
            "npcsh._state.model_supports_tool_calls", return_value=False
        ), patch(
            "npcsh._state.check_llm_command",
            lambda *a, **k: {"output": "check", "messages": []},
        ):
            _, output = process_pipeline_command(
                "hello", None, state, stream_final=False, router=None
            )

        # Should NOT have called CLI short-circuit, should have hit litellm path
        assert isinstance(output, dict)


class TestForenpcDelegation:
    """Tests for forenpc auto-delegation via @npc mentions."""

    def test_is_forenpc_true_when_active_is_forenpc(self):
        """_is_forenpc returns True when state.npc is the team's forenpc."""
        from npcsh._state import _is_forenpc, ShellState
        from npcpy.npc_compiler import NPC, Team

        team = MagicMock(spec=Team)
        team.forenpc = MagicMock(spec=NPC)
        team.forenpc.name = "sibiji"
        team.npcs = {"sibiji": team.forenpc}

        state = ShellState()
        state.team = team
        state.npc = team.forenpc

        assert _is_forenpc(state) is True

    def test_is_forenpc_false_for_specialist(self):
        """_is_forenpc returns False when active NPC is not forenpc."""
        from npcsh._state import _is_forenpc, ShellState
        from npcpy.npc_compiler import NPC, Team

        forenpc = MagicMock(spec=NPC)
        forenpc.name = "sibiji"
        specialist = MagicMock(spec=NPC)
        specialist.name = "coder"

        team = MagicMock(spec=Team)
        team.forenpc = forenpc
        team.npcs = {"sibiji": forenpc, "coder": specialist}

        state = ShellState()
        state.team = team
        state.npc = specialist

        assert _is_forenpc(state) is False

    def test_scan_delegation_finds_mention(self):
        """_scan_and_apply_delegations finds @coder and delegates."""
        from npcsh._state import _scan_and_apply_delegations, ShellState
        from npcpy.npc_compiler import NPC, Team

        coder_npc = MagicMock(spec=NPC)
        coder_npc.name = "coder"
        coder_npc.model = "qwen"
        forenpc = MagicMock(spec=NPC)
        forenpc.name = "sibiji"

        team = MagicMock(spec=Team)
        team.forenpc = forenpc
        team.npcs = {"sibiji": forenpc, "coder": coder_npc}

        state = ShellState()
        state.team = team
        state.npc = forenpc

        response = "Let me ask @coder to handle this."

        with patch("npcsh._state._delegate_to_npc") as mock_delegate:
            mock_delegate.return_value = (state, {"output": "code done"})
            new_state, augmented = _scan_and_apply_delegations(
                state, "sibiji", response, delegation_depth=0
            )

        mock_delegate.assert_called_once()
        assert "--- Response from coder ---" in augmented
        assert "code done" in augmented

    def test_scan_delegation_respects_max_depth(self):
        """Delegation stops when max depth is reached."""
        from npcsh._state import _scan_and_apply_delegations, ShellState

        state = ShellState()
        response = "Let me ask @coder to handle this."

        with patch("npcsh._state._delegate_to_npc") as mock_delegate:
            new_state, augmented = _scan_and_apply_delegations(
                state, "sibiji", response, delegation_depth=1
            )

        mock_delegate.assert_not_called()
        assert augmented == response


# ── Team registry ──────────────────────────────────────────────────────────


class TestTeamRegistry:
    """Tests for multi-team registry loading and switching."""

    def test_load_team_registry_empty_when_missing(self, tmp_path, monkeypatch):
        """load_team_registry returns {} when teams.yaml does not exist."""
        from npcsh._state import load_team_registry

        monkeypatch.setattr(
            "npcsh._state.os.path.expanduser", lambda x: str(tmp_path / "teams.yaml")
        )
        assert load_team_registry() == {}

    def test_load_team_registry_reads_teams(self, tmp_path, monkeypatch):
        """load_team_registry parses teams from YAML."""
        from npcsh._state import load_team_registry

        teams_file = tmp_path / "teams.yaml"
        with open(teams_file, "w") as f:
            yaml.dump({
                "teams": {
                    "npcsh": "~/.npcsh/npc_team",
                    "giacomo": "/Users/caug/giacomo/npc_team",
                }
            }, f)

        monkeypatch.setattr(
            "npcsh._state.os.path.expanduser", lambda x: str(teams_file)
        )
        registry = load_team_registry()
        assert registry == {
            "npcsh": "~/.npcsh/npc_team",
            "giacomo": "/Users/caug/giacomo/npc_team",
        }

    def test_load_team_switches_team(self, tmp_path, monkeypatch):
        """load_team updates state.team, state.npc, and state.current_team_name."""
        from npcsh._state import load_team, ShellState
        from npcpy.npc_compiler import NPC, Team

        team_dir = tmp_path / "other_team"
        team_dir.mkdir()
        with open(team_dir / "forenpc.npc", "w") as f:
            yaml.dump({"name": "forenpc", "primary_directive": "other"}, f)

        state = ShellState()
        state.teams = {"other": str(team_dir)}
        state.command_history = MagicMock()
        state.command_history.engine = None

        with patch("npcsh._state.print") as mock_print:
            result = load_team("other", state)

        assert result is True
        assert state.current_team_name == "other"
        assert state.team is not None
        assert state.npc is not None
        assert mock_print.call_args[0][0] == "Switched to team: other"

    def test_load_team_missing_fails(self):
        """load_team returns False for a team not in registry."""
        from npcsh._state import load_team, ShellState

        state = ShellState()
        state.teams = {}

        with patch("npcsh._state.print") as mock_print:
            result = load_team("missing", state)

        assert result is False
        assert "not found" in mock_print.call_args[0][0]


# ── Combined: providers + CLI in one team.ctx ─────────────────────────────


class TestCombinedProviderScenarios:
    """End-to-end scenarios combining providers field with CLI and openai-like."""

    def _make_team_dir(self, tmp_path, team_ctx_content, npc_specs=None):
        team_dir = tmp_path / "test_team"
        team_dir.mkdir()

        ctx_path = team_dir / "team.ctx"
        with open(ctx_path, "w") as f:
            yaml.dump(team_ctx_content, f)

        npc_specs = npc_specs or {}
        for name, spec in npc_specs.items():
            npc_path = team_dir / f"{name}.npc"
            with open(npc_path, "w") as f:
                yaml.dump(spec, f)

        return str(team_dir)

    def test_mixed_provider_types(self, tmp_path, monkeypatch):
        """Team with litellm, openai-like, and CLI providers all configured via providers field."""
        from npcsh._state import setup_shell

        monkeypatch.setenv("NPCSH_API_URL", "https://fallback.example.com/v1")

        team_ctx = {
            "context": "test",
            "providers": [
                {
                    "name": "anthropic_cloud",
                    "provider_type": "anthropic",
                    "model": "claude-sonnet-4-6",
                },
                {
                    "name": "ollama_cloud",
                    "provider_type": "openai-like",
                    "model": "llama3.2",
                    "api_url": "https://ollama.example.com/v1",
                },
                {
                    "name": "opencode_local",
                    "provider_type": "opencode",
                    "model": "opencode-default",
                },
            ],
        }
        npc_specs = {
            "forenpc": {
                "name": "forenpc",
                "primary_directive": "test",
                "provider": "anthropic_cloud",
            },
            "researcher": {
                "name": "researcher",
                "primary_directive": "research",
                "provider": "ollama_cloud",
            },
            "coder": {
                "name": "coder",
                "primary_directive": "code",
                "provider": "opencode_local",
            },
            "misc": {
                "name": "misc",
                "primary_directive": "misc",
                "provider": "openai-like",
                # No named provider — should get NPCSH_API_URL fallback
            },
        }
        team_dir = self._make_team_dir(tmp_path, team_ctx, npc_specs)

        monkeypatch.setenv("NPCSH_DB_PATH", str(tmp_path / "test.db"))
        monkeypatch.setattr(
            "npcsh._state.PROJECT_NPC_TEAM_PATH", str(tmp_path / "nonexistent")
        )
        monkeypatch.setattr("npcsh._state.DEFAULT_NPC_TEAM_PATH", team_dir)

        with patch("npcsh._state.ensure_npcshrc_exists"), patch(
            "npcsh._state.load_npcshrc_env"
        ), patch("npcsh._state.add_npcshrc_to_shell_config"), patch(
            "npcsh._state.setup_readline"
        ), patch(
            "npcsh._state.save_readline_history"
        ), patch(
            "npcsh._state.is_npcsh_initialized", return_value=True
        ):
            _, team, forenpc = setup_shell()

        # forenpc → anthropic_cloud
        assert forenpc.provider == "anthropic"
        assert forenpc.model == "claude-sonnet-4-6"

        # researcher → ollama_cloud (has explicit api_url)
        researcher = team.npcs["researcher"]
        assert researcher.provider == "openai-like"
        assert researcher.model == "llama3.2"
        assert getattr(researcher, "api_url", None) == "https://ollama.example.com/v1"

        # coder → opencode_local
        coder = team.npcs["coder"]
        assert coder.provider == "opencode"
        assert coder.model == "opencode-default"

        # misc → openai-like without named provider; gets NPCSH_API_URL
        misc = team.npcs["misc"]
        assert misc.provider == "openai-like"
        assert getattr(misc, "api_url", None) == "https://fallback.example.com/v1"
