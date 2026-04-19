import time

duration = float(context.get('duration', 1))
messages = context.get('messages', [])

try:
    time.sleep(duration)
    context['output'] = f"Waited {duration} seconds"
except Exception as e:
    context['output'] = f"Wait failed: {e}"

context['messages'] = messages
