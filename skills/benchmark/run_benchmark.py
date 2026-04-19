import sys

model = ({{ model | default("") | tojson }}).strip()
provider = ({{ provider | default("") | tojson }}).strip()
category = ({{ category | default("") | tojson }}).strip() or None
difficulty = ({{ difficulty | default("") | tojson }}).strip() or None
task_id_filter = ({{ task_id | default("") | tojson }}).strip() or None
timeout = int(({{ timeout | default("120") | tojson }}).strip() or "120")

if not model:
    model = npc.model if npc and npc.model else state.chat_model if state else ""
if not provider:
    provider = npc.provider if npc and npc.provider else state.chat_provider if state else ""

if not model or not provider:
    context['output'] = "Error: model and provider are required. Usage: /benchmark model=qwen3:4b provider=ollama"
else:
    try:
        from npcsh.benchmark.local_runner import run_benchmark

        report = run_benchmark(
            model=model,
            provider=provider,
            category=category,
            difficulty=difficulty,
            task_id=task_id_filter,
            timeout=timeout,
        )

        lines = []
        lines.append(f"## Benchmark Results: {provider}/{model}")
        lines.append("")
        lines.append(f"**Total:** {report.total}")
        lines.append(f"**Passed:** {report.passed}")
        lines.append(f"**Failed:** {report.failed}")
        lines.append(f"**Errors:** {report.errors}")
        pct = (100 * report.passed / report.total) if report.total > 0 else 0
        lines.append(f"**Pass Rate:** {pct:.1f}%")
        lines.append(f"**Duration:** {report.duration:.1f}s")
        lines.append("")

        if report.by_category:
            lines.append("### By Category")
            for cat, stats in sorted(report.by_category.items()):
                p = stats.get("passed", 0)
                t = stats.get("total", 0)
                cat_pct = (100 * p / t) if t > 0 else 0
                lines.append(f"- **{cat}:** {p}/{t} ({cat_pct:.0f}%)")
            lines.append("")

        if report.by_difficulty:
            lines.append("### By Difficulty")
            for diff, stats in sorted(report.by_difficulty.items()):
                p = stats.get("passed", 0)
                t = stats.get("total", 0)
                diff_pct = (100 * p / t) if t > 0 else 0
                lines.append(f"- **{diff}:** {p}/{t} ({diff_pct:.0f}%)")
            lines.append("")

        failed_tasks = [r for r in report.results if not r.passed]
        if failed_tasks:
            lines.append("### Failed Tasks")
            for r in failed_tasks[:20]:
                err = r.error or "verification failed"
                lines.append(f"- **{r.task_id}** ({r.category}): {err}")

        context['output'] = "\n".join(lines)

    except Exception as e:
        import traceback
        context['output'] = f"Error running benchmark: {e}\n\n{traceback.format_exc()}"
