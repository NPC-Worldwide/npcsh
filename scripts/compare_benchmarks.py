#!/usr/bin/env python3
"""
compare_benchmarks.py

Compare two benchmark CSV files and produce a delta report.

Usage:
    python scripts/compare_benchmarks.py baseline.csv new.csv
"""

import argparse
import csv


def load_results(path: str):
    results = {}
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results[row["task_id"]] = {
                "passed": row.get("passed", "").lower() == "true",
                "category": row.get("category", ""),
                "difficulty": row.get("difficulty", ""),
                "duration": float(row.get("duration", "0") or 0),
                "attempts": int(row.get("attempts", "1") or 1),
            }
    return results


def compare(baseline: dict, new: dict):
    all_tasks = sorted(set(baseline.keys()) | set(new.keys()))
    improved = []
    regressed = []
    unchanged_pass = []
    unchanged_fail = []
    new_only = []
    missing = []

    for tid in all_tasks:
        b = baseline.get(tid)
        n = new.get(tid)
        if b and n:
            if b["passed"] and not n["passed"]:
                regressed.append(tid)
            elif not b["passed"] and n["passed"]:
                improved.append(tid)
            elif b["passed"] and n["passed"]:
                unchanged_pass.append(tid)
            else:
                unchanged_fail.append(tid)
        elif n and not b:
            new_only.append(tid)
        elif b and not n:
            missing.append(tid)

    return {
        "improved": improved,
        "regressed": regressed,
        "unchanged_pass": unchanged_pass,
        "unchanged_fail": unchanged_fail,
        "new_only": new_only,
        "missing": missing,
    }


def category_breakdown(results: dict, task_ids: list):
    cats = {}
    for tid in task_ids:
        cat = results.get(tid, {}).get("category", "unknown")
        cats.setdefault(cat, []).append(tid)
    return cats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", help="Baseline benchmark CSV")
    parser.add_argument("new", help="New benchmark CSV")
    args = parser.parse_args()

    baseline = load_results(args.baseline)
    new = load_results(args.new)

    print(f"Baseline: {len(baseline)} tasks")
    print(f"New:      {len(new)} tasks")
    print()

    base_pass = sum(1 for r in baseline.values() if r["passed"])
    new_pass = sum(1 for r in new.values() if r["passed"])
    print(f"Baseline passed: {base_pass}/{len(baseline)} ({100*base_pass/len(baseline):.0f}%)")
    print(f"New passed:      {new_pass}/{len(new)} ({100*new_pass/len(new):.0f}%)")
    print(f"Delta:           {new_pass - base_pass:+d}")
    print()

    comp = compare(baseline, new)
    print(f"Improved:   {len(comp['improved'])}")
    print(f"Regressed:  {len(comp['regressed'])}")
    print(f"New tasks:  {len(comp['new_only'])}")
    print(f"Missing:    {len(comp['missing'])}")
    print()

    if comp["improved"]:
        print("Improved tasks:")
        cats = category_breakdown(new, comp["improved"])
        for cat, tasks in sorted(cats.items()):
            print(f"  {cat}: {', '.join(tasks)}")
        print()

    if comp["regressed"]:
        print("Regressed tasks:")
        cats = category_breakdown(baseline, comp["regressed"])
        for cat, tasks in sorted(cats.items()):
            print(f"  {cat}: {', '.join(tasks)}")
        print()

    # Duration comparison for common passed tasks
    common_pass = [tid for tid in baseline if tid in new and baseline[tid]["passed"] and new[tid]["passed"]]
    if common_pass:
        base_dur = sum(baseline[tid]["duration"] for tid in common_pass) / len(common_pass)
        new_dur = sum(new[tid]["duration"] for tid in common_pass) / len(common_pass)
        print(f"Avg duration (common passed): baseline={base_dur:.1f}s new={new_dur:.1f}s delta={new_dur-base_dur:+.1f}s")


if __name__ == "__main__":
    main()
