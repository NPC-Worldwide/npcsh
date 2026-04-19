from npcpy.work.desktop import perform_action

key = context.get('key', 'enter')
messages = context.get('messages', [])

try:
    perform_action({'type': 'key', 'keys': key})
    context['output'] = "Pressed key: " + str(key)
except Exception as e:
    context['output'] = "Key press failed: " + str(e)

context['messages'] = messages
