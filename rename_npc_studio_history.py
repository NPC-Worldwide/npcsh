#!/usr/bin/env python3
"""Replace 'npc-studio' with 'incognidev' in all path columns of npcsh_history.db."""

import sqlite3
import os

DB = os.path.expanduser("~/npcsh_history.db")

UPDATES = [
    ("command_history", "location"),
    ("command_history", "output"),
    ("conversation_history", "directory_path"),
    ("compiled_npcs", "source_path"),
    ("file_analysis_states", "file_path"),
    ("message_attachments", "file_path"),
    ("npc_versions", "team_path"),
    ("kg_concepts", "directory_path"),
    ("kg_facts", "directory_path"),
    ("kg_fact_sources", "directory_path"),
    ("kg_links", "directory_path"),
    ("kg_metadata", "directory_path"),
    ("memory_lifecycle", "directory_path"),
    ("bookmarks", "folder_path"),
    ("browser_history", "folder_path"),
    ("browser_navigations", "folder_path"),
    ("site_limits", "folder_path"),
    ("pdf_drawings", "file_path"),
    ("pdf_drawings", "svg_path"),
    ("pdf_highlights", "file_path"),
    ("plot_states", "figure_path"),
    ("alicanto_personas", "location"),
]

conn = sqlite3.connect(DB)
total = 0

for table, col in UPDATES:
    try:
        cur = conn.execute(
            f"UPDATE {table} SET {col} = REPLACE({col}, 'npc-studio', 'incognidev') "
            f"WHERE {col} LIKE '%npc-studio%'"
        )
        if cur.rowcount:
            print(f"  {table}.{col}: {cur.rowcount} rows updated")
            total += cur.rowcount
    except sqlite3.OperationalError:
        pass  # table/column doesn't exist in this db version

conn.commit()
conn.close()
print(f"\nDone. {total} rows updated.")
