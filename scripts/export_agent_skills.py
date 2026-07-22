#!/usr/bin/env python3
from pathlib import Path
import re
import sys
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPO_ROOT / "npcsh" / "npc_team" / "jinxes" / "skills"
OUTPUT_DIR = REPO_ROOT / "skills"


def parse_jinx(path):
    text = path.read_text(encoding="utf-8")
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) < 3:
            raise ValueError(f"{path} malformed frontmatter")
        frontmatter = yaml.safe_load(parts[1].strip()) or {}
        data = yaml.safe_load(parts[2].strip()) or {}
    else:
        data = yaml.safe_load(text) or {}
        frontmatter = data
    return {
        "frontmatter": frontmatter,
        "data": data,
        "source": str(path.relative_to(REPO_ROOT)),
    }


def extract_sections(data):
    steps = data.get("steps") or []
    if not steps:
        return {}
    first_step = steps[0]
    return first_step.get("sections") or {}


def render_skill(parsed):
    fm = parsed["frontmatter"]
    sections = extract_sections(parsed["data"])
    name = fm.get("skill_name") or fm.get("jinx_name") or "unnamed-skill"
    lines = [
        "---",
        f"name: {name}",
        f"description: {fm.get('description', '')}",
        f"source_jinx: {parsed['source']}",
        f"engine: {fm.get('engine', 'skill')}",
        "---",
        "",
        f"# {name}",
        "",
        fm.get("description", ""),
        "",
    ]
    for section_name, section_body in sections.items():
        lines.append(f"## {section_name.replace('_', ' ').title()}")
        lines.append("")
        lines.append(section_body)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def export_skill(jinx_path):
    parsed = parse_jinx(jinx_path)
    name = parsed["frontmatter"].get("skill_name") or parsed["frontmatter"].get("jinx_name")
    if not name:
        raise ValueError(f"{jinx_path} missing skill name")
    skill_dir = OUTPUT_DIR / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_path = skill_dir / "SKILL.md"
    skill_path.write_text(render_skill(parsed), encoding="utf-8")
    return skill_path


def main():
    if not SOURCE_DIR.exists():
        print(f"missing source dir: {SOURCE_DIR}", file=sys.stderr)
        return 1
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    exported = []
    for jinx_path in sorted(SOURCE_DIR.glob("*.jinx")):
        try:
            out = export_skill(jinx_path)
            exported.append(out)
            print(f"exported {jinx_path.name} -> {out.relative_to(REPO_ROOT)}")
        except Exception as exc:
            print(f"failed {jinx_path}: {exc}", file=sys.stderr)
            return 1
    current = {p.stem for p in SOURCE_DIR.glob("*.jinx")}
    import shutil
    for skill_dir in OUTPUT_DIR.iterdir():
        if not skill_dir.is_dir() or skill_dir.name in current:
            continue
        marker = skill_dir / "SKILL.md"
        if marker.exists() and "source_jinx:" in marker.read_text(encoding="utf-8"):
            print(f"removing stale {skill_dir.relative_to(REPO_ROOT)}")
            shutil.rmtree(skill_dir)
    print(f"exported {len(exported)} skill(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
