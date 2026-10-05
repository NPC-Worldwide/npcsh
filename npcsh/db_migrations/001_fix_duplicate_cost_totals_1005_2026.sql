PRAGMA journal_mode = MEMORY;
PRAGMA synchronous = NORMAL;
PRAGMA temp_store = MEMORY;
PRAGMA cache_size = -100000;

CREATE TABLE IF NOT EXISTS schema_migrations (
    name TEXT PRIMARY KEY,
    applied_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

DELETE FROM conversation_history
WHERE message_id IS NULL OR message_id = '';

DELETE FROM conversation_history AS c1
WHERE c1.role = 'assistant'
  AND c1.conversation_id IS NOT NULL
  AND c1.conversation_id != ''
  AND EXISTS (
      SELECT 1
      FROM conversation_history AS c2
      WHERE c2.role = 'assistant'
        AND c2.conversation_id = c1.conversation_id
        AND COALESCE(c2.content, '') = COALESCE(c1.content, '')
        AND c2.id < c1.id
  );

PRAGMA synchronous = FULL;
