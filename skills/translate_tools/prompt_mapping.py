import json

raw = {{ foreign_tools | tojson }}
form_title = {{ title | tojson }}

try:
    tools = json.loads(raw) if isinstance(raw, str) else raw
except (json.JSONDecodeError, TypeError):
    tools = []

tools = [t for t in tools if isinstance(t, str) and t.strip()]

if not tools:
    context['output'] = json.dumps({})
else:
    available = []
    team = getattr(npc, 'team', None) if npc else None
    if team is None and state is not None:
        team = getattr(state, 'team', None)
    if team is not None and hasattr(team, 'jinxes_dict'):
        available = sorted(team.jinxes_dict.keys())

    options = ['skip'] + available
    fields = [
        {
            "name": t,
            "type": "select",
            "label": "Map '" + t + "' to",
            "options": options,
            "default": "skip",
            "required": True,
        }
        for t in tools
    ]

    ask_form_jinx = None
    if team is not None and hasattr(team, 'jinxes_dict'):
        ask_form_jinx = team.jinxes_dict.get('ask_form')

    if ask_form_jinx is None:
        context['output'] = json.dumps({
            "error": "ask_form jinx is not available on this team; cannot prompt."
        })
    else:
        ask_ctx = ask_form_jinx.execute(
            input_values={
                "title": form_title,
                "fields": json.dumps(fields),
            },
            npc=npc,
            messages=list(messages) if messages else [],
        )
        ask_output = ask_ctx.get('output', '{}') if isinstance(ask_ctx, dict) else ask_ctx
        try:
            mapping = json.loads(ask_output) if isinstance(ask_output, str) else ask_output
        except (json.JSONDecodeError, TypeError):
            mapping = {}
        if not isinstance(mapping, dict):
            mapping = {}
        if mapping.get('cancelled'):
            context['output'] = json.dumps({"cancelled": True})
        else:
            final = {
                k: v for k, v in mapping.items()
                if isinstance(v, str) and v and v != 'skip'
            }
            context['output'] = json.dumps(final)
