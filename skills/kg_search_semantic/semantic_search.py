import os
from sqlalchemy import create_engine
from npcpy.memory.knowledge_graph import find_similar_facts_chroma, _sync_kg_facts_to_chroma
from npcpy.gen.embeddings import get_embeddings

_q = (context.get('query') or '').strip()
_limit = int(context.get('limit', 10) or 10)
_emb_model = context.get('embedding_model') or 'nomic-embed-text'
_emb_provider = context.get('embedding_provider') or 'ollama'

if not _q:
    output = "Error: query is required"
else:
    _db = os.getenv('NPCSH_DB_PATH', os.path.expanduser('~/npcsh_history.db'))
    _engine = create_engine('sqlite:///' + _db)
    _chroma_path = os.path.join(os.path.expanduser('~'), '.npcsh', 'chroma')

    try:
        _client, _coll = _sync_kg_facts_to_chroma(
            _engine, _chroma_path,
            embedding_model=_emb_model,
            embedding_provider=_emb_provider,
        )
        if not _coll:
            output = "Error: could not sync KG facts to embedding store. Run /kg embed first."
        else:
            _query_emb = get_embeddings([_q], _emb_model, _emb_provider)[0]
            _results = find_similar_facts_chroma(_coll, _q, _query_emb, n_results=_limit)
            import json as _json
            output = _json.dumps({'facts': _results}, indent=2)
    except Exception as _e:
        output = f"Error: {_e}"
