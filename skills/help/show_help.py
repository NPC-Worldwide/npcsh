import json
from pathlib import Path

topic = context.get('topic')

if topic:
    jinx_obj = None
    if hasattr(npc, 'team') and npc.team and hasattr(npc.team, 'jinxes_dict'):
        jinx_obj = npc.team.jinxes_dict.get(topic)
    if not jinx_obj and hasattr(npc, 'jinxes_dict'):
        jinx_obj = npc.jinxes_dict.get(topic)

    if jinx_obj:
        output = f"## /{topic}\n\n"
        output += f"{jinx_obj.description}\n\n"
        if hasattr(jinx_obj, 'inputs') and jinx_obj.inputs:
            output += "**Inputs:**\n"
            for inp in jinx_obj.inputs:
                if isinstance(inp, dict):
                    for k, v in inp.items():
                        default = f" (default: {v})" if v else ""
                        output += f"  - `{k}`{default}\n"
                else:
                    output += f"  - `{inp}`\n"
    else:
        # Check if it's an NPC name
        npc_obj = None
        if hasattr(npc, 'team') and npc.team:
            npc_obj = npc.team.npcs.get(topic)
        if npc_obj:
            output = f"## @{topic}\n\n"
            output += f"**Model:** {getattr(npc_obj, 'model', 'default')}\n"
            directive = getattr(npc_obj, 'primary_directive', '')
            if directive:
                output += f"\n{directive}\n"
            jinxes = getattr(npc_obj, 'jinxes', [])
            if jinxes:
                output += f"\n**Jinxes:** {', '.join(str(j).split('/')[-1] for j in jinxes[:15])}"
                if len(jinxes) > 15:
                    output += f" (+{len(jinxes) - 15} more)"
                output += "\n"
        else:
            output = f"No help found for `{topic}`. Try `/help` for an overview."
else:
    lines = []
    lines.append("# npcsh\n\n")

    lines.append("## Basics\n\n")
    lines.append("  Just type naturally — npcsh figures out what to do.\n")
    lines.append("  Use `@npc_name` to talk to a specific NPC.\n")
    lines.append("  Use `|` to pipe: `git diff | review this`\n")
    lines.append("  Use `/command` for jinxes (tools).\n\n")

    # Gather all jinxes, grouped by directory category
    all_jinxes = {}
    if hasattr(npc, 'team') and npc.team and hasattr(npc.team, 'jinxes_dict'):
        all_jinxes.update(npc.team.jinxes_dict)
    if hasattr(npc, 'jinxes_dict') and npc.jinxes_dict:
        all_jinxes.update(npc.jinxes_dict)

    # Group by FHS category using the jinx's source path
    categories = {
        'Interactive Modes': [],
        'Tools': [],
        'Settings': [],
        'System': [],
        'Other': [],
    }

    for name in sorted(all_jinxes.keys()):
        jinx_obj = all_jinxes[name]
        desc = getattr(jinx_obj, 'description', '')
        # Truncate long descriptions
        if len(desc) > 60:
            desc = desc[:57] + "..."
        entry = f"  /{name:<18} {desc}\n"

        # Categorize by source path
        src = str(getattr(jinx_obj, '_source_path', '') or '')
        if '/bin/' in src or '/usr/' in src:
            categories['Interactive Modes'].append(entry)
        elif '/etc/' in src:
            categories['Settings'].append(entry)
        elif '/sys/' in src:
            categories['System'].append(entry)
        elif '/lib/' in src:
            categories['Tools'].append(entry)
        else:
            categories['Other'].append(entry)

    for cat_name, entries in categories.items():
        if entries:
            lines.append(f"## {cat_name}\n\n")
            lines.extend(entries)
            lines.append("\n")

    # Show team NPCs
    if hasattr(npc, 'team') and npc.team and hasattr(npc.team, 'npcs'):
        team_npcs = npc.team.npcs
        if team_npcs:
            lines.append("## Your Team\n\n")
            for npc_name in sorted(team_npcs.keys()):
                npc_member = team_npcs[npc_name]
                model = getattr(npc_member, 'model', '')
                directive = getattr(npc_member, 'primary_directive', '')
                # First sentence only
                short = directive.split('.')[0].strip() if directive else 'No description'
                if len(short) > 55:
                    short = short[:52] + "..."
                lines.append(f"  @{npc_name:<16} {short}\n")
            lines.append("\n")

    lines.append("Run `/help <command>` or `/help <npc>` for details.\n")

    output = "".join(lines)
