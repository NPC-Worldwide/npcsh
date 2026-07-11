"""CLI runner for jinx-level tests and benchmarks.

Usage:
    python -m npcsh.benchmark.jinx_runner
    python -m npcsh.benchmark.jinx_runner --team ~/.npcsh/npc_team
    python -m npcsh.benchmark.jinx_runner --jinx shell
    python -m npcsh.benchmark.jinx_runner --integration
"""

import argparse
import os
import sys
from pathlib import Path


def _default_team_dir() -> str:
    cwd = Path.cwd()
    if (cwd / "npc_team" / "jinxes").is_dir():
        return str(cwd / "npc_team")
    if (cwd / ".npcsh_team" / "jinxes").is_dir():
        return str(cwd / ".npcsh_team")
    return os.path.expanduser("~/.npcsh/npc_team")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run jinx-level tests and benchmarks")
    parser.add_argument(
        "--team",
        default=_default_team_dir(),
        help="Path to the npc_team directory to test (default: nearest npc_team or ~/.npcsh/npc_team)",
    )
    parser.add_argument(
        "--jinx",
        default=None,
        help="Run only tests for the named jinx",
    )
    parser.add_argument(
        "--test",
        default=None,
        help="Run only the test with this id",
    )
    parser.add_argument(
        "--integration",
        action="store_true",
        help="Run npc_usage integration tests that require a live model",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for CSV output (default: ~/.npcsh/benchmarks/jinxes/)",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Skip writing the CSV report",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full output for every test",
    )
    args = parser.parse_args(argv)

    team_dir = Path(args.team).expanduser()
    if not (team_dir / "jinxes").is_dir():
        print(f"Error: no jinxes directory found under {team_dir}", file=sys.stderr)
        return 1

    from npcsh.jinx_tester import run_all_tests, print_report, write_report_csv

    report = run_all_tests(
        team_dir=str(team_dir),
        integration=args.integration,
        jinx_filter=args.jinx,
        test_filter=args.test,
    )

    print_report(report)

    if args.verbose:
        print()
        print("Detailed results:")
        for r in report.results:
            status = "PASS" if r.passed else "FAIL"
            print(f"\n[{status}] {r.jinx_name}::{r.test_id} ({r.kind}/{r.test_type}) {r.duration_ms:.1f}ms")
            if r.output:
                print(r.output[:2000])
            if r.error:
                print(f"Error: {r.error}")

    if not args.no_csv:
        out_path = write_report_csv(report, output_dir=args.output_dir)
        print(f"CSV written: {out_path}")

    return 0 if report.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
