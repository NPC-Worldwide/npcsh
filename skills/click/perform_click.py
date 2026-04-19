from npcpy.work.desktop import perform_action

x = float(context.get('x', 50))
y = float(context.get('y', 50))
messages = context.get('messages', [])

try:
    perform_action({'type': 'click', 'x': x, 'y': y})
    context['output'] = f"Clicked at ({x}%, {y}%)"
except Exception as e:
    context['output'] = f"Click failed: {e}"

context['messages'] = messages
