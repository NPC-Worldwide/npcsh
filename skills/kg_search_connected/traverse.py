import os
from sqlalchemy import text, create_engine
from collections import deque

_seed = (context.get('seed') or '').strip()
_max_depth = int(context.get('max_depth', 2) or 2)
_max_per_hop = int(context.get('max_per_hop', 10) or 10)

if not _seed:
    output = "Error: seed (a fact statement or concept name) is required"
else:
    _db = os.getenv('NPCSH_DB_PATH', os.path.expanduser('~/npcsh_history.db'))
    _engine = create_engine('sqlite:///' + _db)

    with _engine.connect() as _conn:
        _rows = _conn.execute(text(
            "SELECT source, target, type FROM kg_links"
        )).fetchall()

    _adj = {}
    for _r in _rows:
        _adj.setdefault(_r.source, []).append((_r.target, _r.type, 'out'))
        _adj.setdefault(_r.target, []).append((_r.source, _r.type, 'in'))

    _visited = {_seed: 0}
    _queue = deque([(_seed, 0)])
    _path = []
    while _queue:
        _node, _depth = _queue.popleft()
        if _depth >= _max_depth:
            continue
        _neighbors = _adj.get(_node, [])[:_max_per_hop]
        for _n, _lt, _dir in _neighbors:
            if _n not in _visited:
                _visited[_n] = _depth + 1
                _path.append({'from': _node, 'to': _n, 'type': _lt, 'dir': _dir, 'hop': _depth + 1})
                _queue.append((_n, _depth + 1))

    import json as _json
    output = _json.dumps({
        'seed': _seed,
        'reached': [{'node': n, 'hop': h} for n, h in sorted(_visited.items(), key=lambda x: x[1]) if n != _seed],
        'edges_traversed': _path,
    }, indent=2)
