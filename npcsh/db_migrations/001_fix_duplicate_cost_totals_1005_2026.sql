CREATE TABLE IF NOT EXISTS schema_migrations (
    name TEXT PRIMARY KEY,
    applied_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

DELETE FROM conversation_history
WHERE role = 'user'
  AND input_tokens IS NOT NULL
  AND id IN (
      SELECT c1.id
      FROM conversation_history c1
      JOIN conversation_history c2
        ON c1.conversation_id = c2.conversation_id
       AND c1.role = c2.role
       AND c2.role = 'user'
       AND c1.content = c2.content
       AND c2.input_tokens IS NULL
       AND c1.id > c2.id
  );

DELETE FROM conversation_history
WHERE role = 'assistant'
  AND id IN (
      SELECT c1.id
      FROM conversation_history c1
      JOIN conversation_history c2
        ON c1.conversation_id = c2.conversation_id
       AND c1.role = c2.role
       AND c1.role = 'assistant'
       AND COALESCE(c1.content, '') = COALESCE(c2.content, '')
       AND COALESCE(c1.model, '') = COALESCE(c2.model, '')
       AND COALESCE(c1.provider, '') = COALESCE(c2.provider, '')
       AND COALESCE(c1.npc, '') = COALESCE(c2.npc, '')
       AND COALESCE(c1.input_tokens, 0) = COALESCE(c2.input_tokens, 0)
       AND COALESCE(c1.output_tokens, 0) = COALESCE(c2.output_tokens, 0)
       AND COALESCE(c1.cost, '') = COALESCE(c2.cost, '')
       AND c1.id > c2.id
  );

UPDATE conversation_history
SET input_tokens = NULL,
    output_tokens = NULL,
    cost = NULL
WHERE role = 'assistant'
  AND id NOT IN (
      SELECT MAX(id)
      FROM conversation_history
      WHERE role = 'assistant'
      GROUP BY conversation_id
  );
