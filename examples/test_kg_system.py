#!/usr/bin/env python3
"""
Test script for the Knowledge Graph system.
Tests search, backfill, and evolution functionality.

Run from npc-core root: python -m npcsh.examples.test_kg_system
"""

import os
from sqlalchemy import create_engine, text
from npcpy.memory.command_history import CommandHistory, load_kg_from_db, save_kg_to_db
from npcpy.memory.knowledge_graph import (
    kg_search_facts, kg_list_concepts, kg_get_all_facts,
    kg_link_search, kg_embedding_search, kg_hybrid_search,
    kg_explore_concept, kg_backfill_from_memories, kg_get_stats
)


def print_header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_result(name, result, max_items=5):
    if isinstance(result, list):
        print(f"{name}: {len(result)} items")
        for i, item in enumerate(result[:max_items]):
            if isinstance(item, dict):
                print(f"  {i+1}. [{item.get('type', 'unknown')}] {str(item.get('content', item))[:60]}...")
            else:
                print(f"  {i+1}. {str(item)[:60]}...")
        if len(result) > max_items:
            print(f"  ... and {len(result) - max_items} more")
    elif isinstance(result, dict):
        print(f"{name}:")
        for k, v in result.items():
            print(f"  {k}: {v}")
    else:
        print(f"{name}: {result}")


def test_database_state(engine):
    """Check current database state."""
    print_header("Database State")

    with engine.connect() as conn:
        # Memory stats
        print("\nMemory Lifecycle:")
        result = conn.execute(text('''
            SELECT status, COUNT(*) as cnt
            FROM memory_lifecycle
            GROUP BY status
            ORDER BY cnt DESC
        '''))
        for row in result:
            print(f"  {row.status}: {row.cnt}")

        # KG stats
        print("\nKnowledge Graph:")
        facts = conn.execute(text('SELECT COUNT(*) FROM kg_facts')).scalar() or 0
        concepts = conn.execute(text('SELECT COUNT(*) FROM kg_concepts')).scalar() or 0
        links = conn.execute(text('SELECT COUNT(*) FROM kg_links')).scalar() or 0
        print(f"  Facts: {facts}")
        print(f"  Concepts: {concepts}")
        print(f"  Links: {links}")

        # By scope
        print("\nFacts by Scope:")
        result = conn.execute(text('''
            SELECT npc_name, team_name, COUNT(*) as cnt
            FROM kg_facts
            GROUP BY npc_name, team_name
        '''))
        for row in result:
            print(f"  {row.npc_name}/{row.team_name}: {row.cnt}")


def test_keyword_search(engine):
    """Test keyword-based search."""
    print_header("Keyword Search Tests")

    queries = ["Mao", "Oreo", "cookie", "China", "lesson"]

    for query in queries:
        results = kg_search_facts(engine, query)
        print(f"\nQuery '{query}': {len(results)} results")
        for r in results[:2]:
            print(f"  - {r[:50]}...")


def test_concept_operations(engine):
    """Test concept listing and exploration."""
    print_header("Concept Operations")

    # List all concepts
    concepts = kg_list_concepts(engine)
    print(f"\nAll concepts ({len(concepts)}):")
    for c in concepts:
        print(f"  - {c}")

    # Explore a concept if any exist
    if concepts:
        concept_name = concepts[0]
        print(f"\nExploring concept: {concept_name}")
        result = kg_explore_concept(engine, concept_name)
        print(f"  Direct facts: {len(result['direct_facts'])}")
        print(f"  Related concepts: {result['related_concepts']}")
        print(f"  Extended facts: {len(result['extended_facts'])}")


def test_link_search(engine):
    """Test link traversal search."""
    print_header("Link Traversal Search")

    results = kg_link_search(engine, "Oreo", max_depth=2, breadth_per_step=3)
    print_result("Link search 'Oreo' (depth=2)", results)


def test_hybrid_search(engine):
    """Test hybrid search modes."""
    print_header("Hybrid Search Tests")

    modes = ['keyword', 'keyword+link']
    query = "history"

    for mode in modes:
        results = kg_hybrid_search(engine, query, mode=mode, max_results=5)
        print(f"\nMode '{mode}' for '{query}': {len(results)} results")
        for r in results[:3]:
            src = r.get('source', 'unknown')
            print(f"  [{r['type']} {r['score']:.2f} {src}] {r['content'][:45]}...")


def test_backfill_dry_run(engine):
    """Test backfill in dry-run mode."""
    print_header("Backfill Dry Run")

    stats = kg_backfill_from_memories(engine, dry_run=True)
    print(f"Scopes to process: {stats['scopes_processed']}")
    print(f"Current facts: {stats['facts_before']}")
    print(f"Current concepts: {stats['concepts_before']}")

    for s in stats.get('scopes', []):
        scope = s['scope']
        print(f"  {scope[0]}/{scope[1]}: {s['memory_count']} memories")


def test_kg_stats(engine):
    """Test KG stats function."""
    print_header("KG Stats")

    # This tests scope-specific stats
    stats = kg_get_stats(engine)
    print_result("Stats for current scope", stats)


def main():
    db_path = os.path.expanduser("~/npcsh_history.db")

    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}")
        return 1

    engine = create_engine(f'sqlite:///{db_path}')

    print("\n" + "="*60)
    print("  NPCSH Knowledge Graph System Tests")
    print("="*60)

    try:
        test_database_state(engine)
        test_keyword_search(engine)
        test_concept_operations(engine)
        test_link_search(engine)
        test_hybrid_search(engine)
        test_backfill_dry_run(engine)
        test_kg_stats(engine)

        print_header("All Tests Completed Successfully")
        return 0

    except Exception as e:
        import traceback
        print(f"\nError: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
