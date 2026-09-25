import os
import tempfile
import uuid
from datetime import datetime

from sqlalchemy import create_engine, text

from npcpy.npc_compiler import Jinx


COMPRESS_JINX_PATH = os.path.join(
    os.path.dirname(__file__), "..", "npcsh", "npc_team", "jinxes", "usr", "compress.jinx"
)


def _mock_breathe(messages, model=None, provider=None, npc=None, context=None, **kwargs):
    return {"output": "Mocked compression summary."}


def test_compress_jinx_does_not_import_outdated_module():
    with open(COMPRESS_JINX_PATH) as f:
        content = f.read()
    assert "npcpy.memory.command_history" not in content


def test_compress_jinx_creates_linked_conversation(monkeypatch):
    import npcpy.llm_funcs

    monkeypatch.setattr(npcpy.llm_funcs, "breathe", _mock_breathe)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_history.db")
        engine = create_engine(f"sqlite:///{db_path}")
        old_cid = "test_old_conversation"
        last_msg_id = str(uuid.uuid4())

        with engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE conversation_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message_id TEXT UNIQUE NOT NULL,
                    timestamp TEXT,
                    role TEXT,
                    content TEXT,
                    conversation_id TEXT,
                    directory_path TEXT,
                    model TEXT,
                    provider TEXT,
                    npc TEXT,
                    team TEXT,
                    reasoning_content TEXT,
                    tool_calls TEXT,
                    tool_results TEXT,
                    parent_message_id TEXT,
                    device_id TEXT,
                    device_name TEXT,
                    params TEXT,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    cost TEXT
                )
            """))
            conn.execute(text("""
                INSERT INTO conversation_history
                (message_id, timestamp, role, content, conversation_id, directory_path)
                VALUES (:message_id, :timestamp, :role, :content, :conversation_id, :directory_path)
            """), {
                "message_id": last_msg_id,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "role": "user",
                "content": "Hello",
                "conversation_id": old_cid,
                "directory_path": tmpdir,
            })

        class MockState:
            conversation_id = old_cid
            chat_model = None
            chat_provider = None

        state = MockState()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        extra_globals = {"state": state}

        monkeypatch.setenv("NPCSH_DB_PATH", db_path)

        jinx = Jinx(jinx_path=COMPRESS_JINX_PATH)
        result = jinx.execute(input_values={}, messages=messages, extra_globals=extra_globals)

        assert "Compressed. New conversation:" in str(result.get("output", ""))
        assert state.conversation_id != old_cid

        with engine.connect() as conn:
            rows = conn.execute(text(
                "SELECT * FROM conversation_history WHERE conversation_id = :cid ORDER BY id ASC"
            ), {"cid": state.conversation_id}).fetchall()

        assert len(rows) == 1
        assert rows[0].role == "system"
        assert rows[0].parent_message_id == last_msg_id
        assert rows[0].npc == "npcsh"
        assert rows[0].team == "npcsh"


def test_compress_jinx_handles_missing_state(monkeypatch):
    import npcpy.llm_funcs

    monkeypatch.setattr(npcpy.llm_funcs, "breathe", _mock_breathe)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_history.db")
        engine = create_engine(f"sqlite:///{db_path}")

        monkeypatch.setenv("NPCSH_DB_PATH", db_path)

        jinx = Jinx(jinx_path=COMPRESS_JINX_PATH)
        result = jinx.execute(input_values={}, messages=[], extra_globals={})

        assert "Compressed. New conversation:" in str(result.get("output", ""))

        with engine.connect() as conn:
            rows = conn.execute(text(
                "SELECT * FROM conversation_history ORDER BY id ASC"
            )).fetchall()

        assert len(rows) == 1
        assert rows[0].role == "system"
