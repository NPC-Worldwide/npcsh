import os
from sqlalchemy import text, create_engine
from npcpy.memory.command_history import load_kg_from_db

_q = (context.get('query') or '').strip().lower()
_limit = int(context.get('limit', 15) or 15)
_type = (context.get('type') or 'both').lower()

if not _q:
    output = "Error: query is required"
else:
    _db = os.getenv('NPCSH_DB_PATH', os.path.expanduser('~/npcsh_history.db'))
    _engine = create_engine('sqlite:///' + _db)
    _kg = load_kg_from_db(_engine, team_name='', npc_name='', directory_path='')

    _words = set(_q.split())
    _facts_out = []
    _concepts_out = []

    if _type in ('facts', 'both'):
        _scored = []
        for f in _kg.get('facts', []):
            _stmt = (f.get('statement', '') or '').lower()
            _s = sum(1 for w in _words if w in _stmt)
            if _s > 0:
                _scored.append((_s, f))
        _scored.sort(key=lambda x: -x[0])
        _facts_out = [{
            'statement': f.get('statement'),
            'source_text': f.get('source_text'),
            'score': s,
        } for s, f in _scored[:_limit]]

    if _type in ('concepts', 'both'):
        _scored = []
        for c in _kg.get('concepts', []):
            _name = (c.get('name', '') or '').lower()
            _s = sum(1 for w in _words if w in _name)
            if _s > 0:
                _scored.append((_s, c))
        _scored.sort(key=lambda x: -x[0])
        _concepts_out = [{
            'name': c.get('name'),
            'score': s,
        } for s, c in _scored[:_limit]]

    import json as _json
    output = _json.dumps({'facts': _facts_out, 'concepts': _concepts_out}, indent=2)
