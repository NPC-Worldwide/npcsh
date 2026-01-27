import os
import tempfile
import pytest

from npcsh._state import set_npcsh_config_value, CONFIG_KEY_MAP


def test_config_key_map_has_common_keys():
    """Verify common shorthand keys exist in CONFIG_KEY_MAP."""
    assert "model" in CONFIG_KEY_MAP
    assert "provider" in CONFIG_KEY_MAP
    assert "chatmodel" in CONFIG_KEY_MAP
    assert "chatprovider" in CONFIG_KEY_MAP


def test_set_npcsh_config_value_sets_env(monkeypatch):
    """Test that set_npcsh_config_value sets environment variable."""
    # Use a temp file for npcshrc to avoid modifying user's config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.npcshrc', delete=False) as f:
        temp_npcshrc = f.name

    monkeypatch.setattr(os.path, 'expanduser', lambda x: temp_npcshrc if x == "~/.npcshrc" else x)

    set_npcsh_config_value("model", "test-model")

    assert os.environ.get("NPCSH_CHAT_MODEL") == "test-model"

    # Clean up
    os.unlink(temp_npcshrc)


def test_set_npcsh_config_value_persists_to_file(monkeypatch):
    """Test that set_npcsh_config_value persists config to file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.npcshrc', delete=False) as f:
        temp_npcshrc = f.name

    original_expanduser = os.path.expanduser
    monkeypatch.setattr(os.path, 'expanduser', lambda x: temp_npcshrc if x == "~/.npcshrc" else original_expanduser(x))

    set_npcsh_config_value("provider", "test-provider")

    with open(temp_npcshrc) as f:
        content = f.read()

    assert 'export NPCSH_CHAT_PROVIDER="test-provider"' in content

    # Clean up
    os.unlink(temp_npcshrc)
