-- NQL Model: Error Digest
-- Identifies and explains recent errors from command history.
-- Run: /nql model=error_digest

{{ config(materialized='table') }}

SELECT
    timestamp,
    command,
    output,
    synthesize(
        'This command failed: {command}. The error output was: {output}. Explain what went wrong and suggest a fix.',
        'sibiji',
        'error_analysis'
    ) as diagnosis
FROM command_history
WHERE output LIKE '%error%' OR output LIKE '%Error%' OR output LIKE '%ERROR%'
ORDER BY timestamp DESC
LIMIT 20
