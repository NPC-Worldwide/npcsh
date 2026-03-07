-- NQL Model: Command Analytics
-- Analyzes recent command history for usage patterns.
-- Run: /nql model=command_analytics

{{ config(materialized='table') }}

SELECT
    DATE(timestamp) as day,
    command_type,
    COUNT(*) as total_commands,
    synthesize(
        'Summarize the usage pattern: {total_commands} commands of type {command_type} on {day}. What does this suggest about workflow?',
        'sibiji',
        'usage_analysis'
    ) as pattern_insight
FROM command_history
WHERE timestamp >= date('now', '-7 days')
GROUP BY DATE(timestamp), command_type
ORDER BY day DESC, total_commands DESC
