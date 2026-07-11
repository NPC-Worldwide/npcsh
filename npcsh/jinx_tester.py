"""Jinx-level test and benchmark harness.

Discovers tests and benchmarks declared inside jinx YAML files, then runs them
without requiring a full LLM benchmark loop. Supports:

- compilation: load + first-pass render + tool-def build
- capability: execute the jinx with fixed inputs and assert output / success / duration
- npc_usage: integration test where an NPC is prompted to invoke the jinx

Example jinx section::

    tests:
      - id: echo_runs
        type: capability
        inputs:
          bash_command: "echo hello"
        expected:
          success: true
          output_contains: "hello"
          max_duration_ms: 5000
"""

from __future__ import annotations

import os
import re
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from npcpy.npc_compiler import Jinx, Team, load_jinxes_from_directory


@dataclass
class JinxTestResult:
    jinx_name: str
    jinx_path: str
    test_id: str
    test_type: str
    kind: str  # "test" or "benchmark"
    passed: bool
    duration_ms: float
    output: str = ""
    error: Optional[str] = None


@dataclass
class JinxTestReport:
    team_dir: str
    total: int = 0
    passed: int = 0
    failed: int = 0
    results: List[JinxTestResult] = field(default_factory=list)
    by_jinx: Dict[str, Dict[str, int]] = field(default_factory=dict)
    by_type: Dict[str, Dict[str, int]] = field(default_factory=dict)


def _normalize_test_list(value: Any) -> List[Dict[str, Any]]:
    """Accept a single test dict, a list of tests, or missing."""
    if value is None:
        return []
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def discover_jinx_tests(team_dir: str) -> List[Tuple[str, str, Dict[str, Any], str]]:
    """Return [(jinx_path, jinx_name, test_def, kind), ...] for all tests/benchmarks."""
    jinxes_dir = Path(team_dir).expanduser() / "jinxes"
    if not jinxes_dir.is_dir():
        return []

    discovered: List[Tuple[str, str, Dict[str, Any], str]] = []
    for jinx_obj in load_jinxes_from_directory(str(jinxes_dir)):
        path = getattr(jinx_obj, "_source_path", "") or ""
        name = getattr(jinx_obj, "jinx_name", "")
        if not name:
            continue

        for test_def in _normalize_test_list(getattr(jinx_obj, "tests", None)):
            discovered.append((path, name, test_def, "test"))
        for bench_def in _normalize_test_list(getattr(jinx_obj, "benchmarks", None)):
            discovered.append((path, name, bench_def, "benchmark"))

    return discovered


def _build_minimal_npc(team: Team) -> Optional[Any]:
    """Return a minimal NPC-like object for jinx execution when no real NPC is needed."""
    if team.npcs:
        return next(iter(team.npcs.values()))
    return None


def run_compilation_test(team: Team, jinx: Jinx) -> JinxTestResult:
    """Verify the jinx loads, renders, and converts to a tool definition."""
    start = time.perf_counter()
    try:
        # First-pass rendering already happened when Team loaded. Validate tool def.
        tool_def = jinx.to_tool_def()
        if not isinstance(tool_def, dict):
            raise ValueError(f"to_tool_def returned {type(tool_def)}, expected dict")
        if tool_def.get("type") != "function":
            raise ValueError("tool_def missing type='function'")
        duration_ms = (time.perf_counter() - start) * 1000
        return JinxTestResult(
            jinx_name=jinx.jinx_name,
            jinx_path=getattr(jinx, "_source_path", ""),
            test_id="compilation",
            test_type="compilation",
            kind="test",
            passed=True,
            duration_ms=duration_ms,
            output="compiled and tool-def built",
        )
    except Exception as exc:
        duration_ms = (time.perf_counter() - start) * 1000
        return JinxTestResult(
            jinx_name=jinx.jinx_name,
            jinx_path=getattr(jinx, "_source_path", ""),
            test_id="compilation",
            test_type="compilation",
            kind="test",
            passed=False,
            duration_ms=duration_ms,
            output="",
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        )


def _check_expected(output: str, context: Dict[str, Any], expected: Dict[str, Any]) -> Tuple[bool, str]:
    """Evaluate expected conditions against a jinx execution result."""
    success_expected = expected.get("success")
    if success_expected is not None:
        actual_success = context.get("success", True)
        # Jinx.execute returns a context dict; success is not explicitly set,
        # so treat absence of an error output as success.
        if isinstance(actual_success, bool) and actual_success != success_expected:
            return False, f"success mismatch: got {actual_success}, expected {success_expected}"
        if "error" in str(context.get("output", "")).lower() and success_expected:
            return False, "expected success but output contains 'error'"

    if "output_contains" in expected:
        needle = str(expected["output_contains"])
        if needle not in output:
            return False, f"output missing {needle!r}"

    if "output_not_contains" in expected:
        needle = str(expected["output_not_contains"])
        if needle in output:
            return False, f"output unexpectedly contains {needle!r}"

    if "output_equals" in expected:
        expected_output = str(expected["output_equals"])
        if output != expected_output:
            return False, f"output mismatch:\n got: {output!r}\n expected: {expected_output!r}"

    if "output_regex" in expected:
        pattern = str(expected["output_regex"])
        if not re.search(pattern, output):
            return False, f"output does not match regex {pattern!r}"

    return True, ""


def run_capability_test(
    jinx: Jinx,
    test_def: Dict[str, Any],
    team: Optional[Team] = None,
) -> JinxTestResult:
    """Execute a jinx with fixed inputs and evaluate expected conditions."""
    test_id = test_def.get("id", "unnamed_capability")
    inputs = test_def.get("inputs", {})
    if inputs is None:
        inputs = {}
    if not isinstance(inputs, dict):
        return JinxTestResult(
            jinx_name=jinx.jinx_name,
            jinx_path=getattr(jinx, "_source_path", ""),
            test_id=test_id,
            test_type=test_def.get("type", "capability"),
            kind=test_def.get("_kind", "test"),
            passed=False,
            duration_ms=0.0,
            error="inputs must be a dict",
        )

    expected = test_def.get("expected", {})
    if expected is None:
        expected = {}

    npc = _build_minimal_npc(team) if team else None
    start = time.perf_counter()
    try:
        context = jinx.execute(input_values=inputs, npc=npc)
        output = str(context.get("output", ""))
        ok, reason = _check_expected(output, context, expected)

        duration_ms = (time.perf_counter() - start) * 1000
        max_ms = expected.get("max_duration_ms")
        if ok and max_ms is not None and duration_ms > float(max_ms):
            ok = False
            reason = f"duration {duration_ms:.1f}ms exceeds max {max_ms}ms"

        return JinxTestResult(
            jinx_name=jinx.jinx_name,
            jinx_path=getattr(jinx, "_source_path", ""),
            test_id=test_id,
            test_type=test_def.get("type", "capability"),
            kind=test_def.get("_kind", "test"),
            passed=ok,
            duration_ms=duration_ms,
            output=output[:2000],
            error=None if ok else reason,
        )
    except Exception as exc:
        duration_ms = (time.perf_counter() - start) * 1000
        return JinxTestResult(
            jinx_name=jinx.jinx_name,
            jinx_path=getattr(jinx, "_source_path", ""),
            test_id=test_id,
            test_type=test_def.get("type", "capability"),
            kind=test_def.get("_kind", "test"),
            passed=False,
            duration_ms=duration_ms,
            output="",
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        )


def run_jinx_test(
    team: Team,
    jinx: Jinx,
    test_def: Dict[str, Any],
    kind: str,
) -> JinxTestResult:
    """Dispatch a single jinx test/benchmark by type."""
    test_def = dict(test_def)
    test_def["_kind"] = kind
    test_type = test_def.get("type", "capability")

    if test_type == "compilation":
        return run_compilation_test(team, jinx)
    if test_type in ("capability",):
        return run_capability_test(jinx, test_def, team=team)
    if test_type == "npc_usage":
        return JinxTestResult(
            jinx_name=jinx.jinx_name,
            jinx_path=getattr(jinx, "_source_path", ""),
            test_id=test_def.get("id", "unnamed_npc_usage"),
            test_type=test_type,
            kind=kind,
            passed=False,
            duration_ms=0.0,
            error="npc_usage tests are not implemented yet; run with --skip-npc-usage",
        )

    return JinxTestResult(
        jinx_name=jinx.jinx_name,
        jinx_path=getattr(jinx, "_source_path", ""),
        test_id=test_def.get("id", "unknown"),
        test_type=test_type,
        kind=kind,
        passed=False,
        duration_ms=0.0,
        error=f"unknown test type: {test_type}",
    )


def run_all_tests(
    team_dir: str,
    integration: bool = False,
    jinx_filter: Optional[str] = None,
    test_filter: Optional[str] = None,
    include_compilation: bool = True,
) -> JinxTestReport:
    """Discover and run all jinx tests/benchmarks for a team."""
    report = JinxTestReport(team_dir=team_dir)

    raw_tests = discover_jinx_tests(team_dir)
    if not raw_tests:
        return report

    # Compile the team once; compilation tests validate each jinx against this loaded team.
    team = Team(team_path=team_dir)

    for jinx_path, jinx_name, test_def, kind in raw_tests:
        if jinx_filter and jinx_filter != jinx_name:
            continue
        test_id = test_def.get("id", "")
        if test_filter and test_filter != test_id:
            continue

        jinx = team.jinxes_dict.get(jinx_name)
        if jinx is None:
            result = JinxTestResult(
                jinx_name=jinx_name,
                jinx_path=jinx_path,
                test_id=test_id or "missing",
                test_type=test_def.get("type", "unknown"),
                kind=kind,
                passed=False,
                duration_ms=0.0,
                error=f"jinx {jinx_name} not found in compiled team",
            )
        else:
            if test_def.get("type") == "npc_usage" and not integration:
                result = JinxTestResult(
                    jinx_name=jinx_name,
                    jinx_path=jinx_path,
                    test_id=test_id,
                    test_type="npc_usage",
                    kind=kind,
                    passed=True,
                    duration_ms=0.0,
                    output="skipped (integration tests disabled)",
                )
            else:
                result = run_jinx_test(team, jinx, test_def, kind)

        report.results.append(result)
        report.total += 1
        if result.passed:
            report.passed += 1
        else:
            report.failed += 1

        if jinx_name not in report.by_jinx:
            report.by_jinx[jinx_name] = {"total": 0, "passed": 0}
        report.by_jinx[jinx_name]["total"] += 1
        if result.passed:
            report.by_jinx[jinx_name]["passed"] += 1

        type_key = f"{kind}:{result.test_type}"
        if type_key not in report.by_type:
            report.by_type[type_key] = {"total": 0, "passed": 0}
        report.by_type[type_key]["total"] += 1
        if result.passed:
            report.by_type[type_key]["passed"] += 1

    return report


def print_report(report: JinxTestReport) -> None:
    """Print a human-readable summary of the test report."""
    print("=" * 60)
    print("JINX TEST RESULTS")
    print("=" * 60)
    print(f"Team: {report.team_dir}")
    print(f"Total: {report.total}  Passed: {report.passed}  Failed: {report.failed}")
    if report.total > 0:
        print(f"Score: {report.passed}/{report.total} ({100 * report.passed / report.total:.0f}%)")
    print()

    if report.by_jinx:
        print("By jinx:")
        for name, stats in sorted(report.by_jinx.items()):
            print(f"  {name:<30} {stats['passed']}/{stats['total']}")
        print()

    if report.by_type:
        print("By type:")
        for t, stats in sorted(report.by_type.items()):
            print(f"  {t:<30} {stats['passed']}/{stats['total']}")
        print()

    failed = [r for r in report.results if not r.passed and "skipped" not in r.output.lower()]
    if failed:
        print("Failed tests:")
        for r in failed[:20]:
            err = r.error or "assertion failed"
            print(f"  {r.jinx_name}::{r.test_id} ({r.kind}/{r.test_type}): {err.splitlines()[0]}")
        print()


def write_report_csv(report: JinxTestReport, output_dir: Optional[str] = None) -> Path:
    """Write the report to a timestamped CSV under ~/.npcsh/benchmarks/jinxes/."""
    if output_dir is None:
        output_dir = Path.home() / ".npcsh" / "benchmarks" / "jinxes"
    else:
        output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"jinx_tests_{ts}.csv"

    df = pd.DataFrame([
        {
            "jinx_name": r.jinx_name,
            "jinx_path": r.jinx_path,
            "test_id": r.test_id,
            "test_type": r.test_type,
            "kind": r.kind,
            "passed": r.passed,
            "duration_ms": round(r.duration_ms, 1),
            "error": r.error or "",
            "output": r.output,
        }
        for r in report.results
    ])
    df.to_csv(out_path, index=False)
    return out_path
