import sqlite3
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter, defaultdict
import networkx as nx
import os

def deep_analyze_kg():
    """Actually analyze the knowledge graph structure and content"""
    
    db_path = os.path.expanduser("~/npcsh_history.db")
    conn = sqlite3.connect(db_path)
    
    # Load all data
    concepts_df = pd.read_sql_query("SELECT * FROM kg_concepts", conn)
    facts_df = pd.read_sql_query("SELECT * FROM kg_facts", conn)
    links_df = pd.read_sql_query("SELECT * FROM kg_links", conn)
    
    print("=" * 80)
    print("DEEP KNOWLEDGE GRAPH ANALYSIS")
    print("=" * 80)
    
    # Analyze concept clusters and themes
    print("\n🧠 CONCEPT ANALYSIS:")
    print("-" * 50)
    
    # Group concepts by common themes/words
    concept_words = []
    for concept in concepts_df['name']:
        words = concept.lower().split()
        concept_words.extend(words)
    
    word_freq = Counter(concept_words)
    print(f"Most common concept words: {dict(list(word_freq.most_common(10)))}")
    
    # Find concept clusters by shared words
    concept_clusters = defaultdict(list)
    for concept in concepts_df['name']:
        for word in concept.lower().split():
            if word_freq[word] > 2:  # Only frequent words
                concept_clusters[word].append(concept)
    
    print("\nConcept clusters (by shared keywords):")
    for word, concepts in sorted(concept_clusters.items(), key=lambda x: len(x[1]), reverse=True)[:8]:
        if len(concepts) > 2:
            print(f"  '{word}': {len(concepts)} concepts - {concepts[:3]}...")
    
    # Analyze fact patterns
    print("\n📋 FACT ANALYSIS:")
    print("-" * 50)
    
    # Extract command patterns from facts
    command_facts = facts_df[facts_df['statement'].str.contains('command|issued|executed', case=False, na=False)]
    print(f"Command-related facts: {len(command_facts)}")
    
    # Extract user actions
    user_actions = facts_df[facts_df['statement'].str.contains('user', case=False, na=False)]
    action_patterns = []
    for statement in user_actions['statement'].head(10):
        if 'issued' in statement or 'executed' in statement:
            # Extract the command part
            parts = statement.split("'")
            if len(parts) >= 2:
                action_patterns.append(parts[1])
    
    print(f"Common user commands: {Counter(action_patterns).most_common(5)}")
    
    # Analyze assistant responses
    assistant_facts = facts_df[facts_df['statement'].str.contains('assistant|processed', case=False, na=False)]
    print(f"Assistant response facts: {len(assistant_facts)}")
    
    # Analyze link network structure
    print("\n🔗 NETWORK ANALYSIS:")
    print("-" * 50)
    
    # Build network graph
    G = nx.Graph()
    
    # Add concept nodes
    for concept in concepts_df['name']:
        G.add_node(concept, type='concept')
    
    # Add fact-to-concept edges
    fact_concept_links = links_df[links_df['type'] == 'fact_to_concept']
    for _, link in fact_concept_links.iterrows():
        if link['target'] in G.nodes:  # Make sure target concept exists
            G.add_edge(link['source'], link['target'], type='fact_to_concept')
    
    print(f"Network nodes: {len(G.nodes)}")
    print(f"Network edges: {len(G.edges)}")
    
    # Find most connected concepts
    if len(G.nodes) > 0:
        concept_degrees = []
        for node in G.nodes:
            if node in concepts_df['name'].values:
                degree = G.degree(node)
                concept_degrees.append((node, degree))
        
        concept_degrees.sort(key=lambda x: x[1], reverse=True)
        print("\nMost connected concepts:")
        for concept, degree in concept_degrees[:10]:
            print(f"  {concept}: {degree} connections")
    
    # Analyze generational evolution
    print("\n⏳ TEMPORAL ANALYSIS:")
    print("-" * 50)
    
    # Concepts by generation
    gen_counts = concepts_df['generation'].value_counts().sort_index()
    print(f"Concept generation spread: {gen_counts.head(10).to_dict()}")
    
    # Latest concepts
    latest_concepts = concepts_df[concepts_df['generation'] >= concepts_df['generation'].max() - 5]
    print(f"\nRecent concepts (last 5 generations): {list(latest_concepts['name'][:8])}")
    
    # Fact evolution
    if 'generation' in facts_df.columns:
        fact_gen_counts = facts_df['generation'].value_counts().sort_index()
        print(f"Recent fact generations: {fact_gen_counts.tail(5).to_dict()}")
    
    # Analyze directory contexts
    print("\n📁 DIRECTORY CONTEXT ANALYSIS:")
    print("-" * 50)
    
    for directory in concepts_df['directory_path'].unique():
        dir_concepts = concepts_df[concepts_df['directory_path'] == directory]
        dir_facts = facts_df[facts_df['directory_path'] == directory]
        print(f"\nDirectory: {directory}")
        print(f"  Concepts: {len(dir_concepts)} | Facts: {len(dir_facts)}")
        print(f"  Sample concepts: {list(dir_concepts['name'].head(5))}")
    
    # Simple visualization - concept connectivity
    plt.figure(figsize=(12, 8))
    
    if len(concept_degrees) > 0:
        concepts, degrees = zip(*concept_degrees[:15])
        
        plt.barh(range(len(concepts)), degrees, color='steelblue')
        plt.yticks(range(len(concepts)), [c[:40] + '...' if len(c) > 40 else c for c in concepts])
        plt.xlabel('Number of Connections')
        plt.title('Most Connected Concepts in Knowledge Graph')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    
    conn.close()

if __name__ == "__main__":
    deep_analyze_kg()