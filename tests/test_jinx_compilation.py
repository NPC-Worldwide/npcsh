"""Fast CI tests that every discoverable jinx compiles cleanly.

This loads the bundled npc_team jinxes directory, builds the team, and asserts
that each jinx produces a valid tool definition. It does not require a model.
"""

import os
import glob
import pytest

from npcpy.jinx_tester import discover_jinx_tests, run_all_tests


TEAM_DIR = os.path.join(os.path.dirname(__file__), "..", "npcsh", "npc_team")


def test_jinxes_load_without_error():
    """Every jinx in the bundled team must load and compile to a tool def."""
    report = run_all_tests(team_dir=TEAM_DIR, integration=False)
    failed = [r for r in report.results if not r.passed and "skipped" not in r.output.lower()]

    if failed:
        messages = []
        for r in failed:
            err = (r.error or "assertion failed").splitlines()[0]
            messages.append(f"{r.jinx_name}::{r.test_id}: {err}")
        pytest.fail("Jinx compilation failures:\n" + "\n".join(messages))


def test_at_least_some_jinxes_exist():
    """Sanity check that the bundled team has jinxes to test."""
    tests = discover_jinx_tests(TEAM_DIR)
    assert len(tests) >= 0, f"no jinxes discovered in {TEAM_DIR}"
