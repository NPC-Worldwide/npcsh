state = context.get('state')
output_messages = context.get('messages', [])

if state:
    result = state.set_log_level("verbose")
    context['output'] = result
else:
    context['output'] = "Error: state not available"

context['messages'] = output_messages
