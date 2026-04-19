n = int(context.get('n', 10))
msgs = []
if state and state.messages:
    msgs = state.messages[-n:]
elif state and getattr(state, 'conversation_id', None):
    try:
        ch = getattr(state, 'command_history', None)
        if ch:
            db_msgs = ch.get_conversations_by_id(state.conversation_id)
            for m in (db_msgs or [])[-n:]:
                r, c = m.get('role',''), m.get('content','')
                if r in ('user', 'assistant') and c:
                    msgs.append(m)
    except Exception:
        pass
if not msgs:
    print("(no messages in current conversation)")
else:
    for m in msgs:
        r, c = m.get('role',''), m.get('content','')
        if r == 'user':
            print(f"\n\033[32m> {c}\033[0m")
        elif r == 'assistant':
            print(f"\n{c}")
context['output'] = ''
