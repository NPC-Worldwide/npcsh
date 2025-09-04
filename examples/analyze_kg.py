import sqlite3
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter, defaultdict
import networkx as nx
import numpy as np
import os
import re
import warnings
warnings.filterwarnings('ignore')

# Set better matplotlib defaults
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 11

def create_clean_network_viz(G, title="Network", node_attr=None, figsize=(18, 14), 
                            min_label_degree=None, max_labels=15):
    """Create clean, readable network visualization"""
    
    if len(G.nodes()) == 0:
        print("Empty graph - skipping visualization")
        return
    
    plt.figure(figsize=figsize)
    
  
    if len(G.nodes()) > 100:
        pos = nx.spring_layout(G, k=4/np.sqrt(len(G.nodes())), iterations=50)
    else:
        pos = nx.spring_layout(G, k=2, iterations=100)
    
  
    degrees = dict(G.degree())
    max_degree = max(degrees.values()) if degrees else 1
    min_size, max_size = 100, 800
    node_sizes = []
    for node in G.nodes():
        size = min_size + (degrees[node] / max_degree) * (max_size - min_size)
        node_sizes.append(size)
    
  
    if node_attr:
        node_colors = [node_attr.get(node, 0) for node in G.nodes()]
        colormap = 'viridis'
    else:
        node_colors = [degrees[node] for node in G.nodes()]
        colormap = 'plasma'
    
  
    if G.is_directed():
        nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.8, arrows=True, 
                              arrowsize=12, edge_color='#666666')
    else:
        if G.edges() and 'weight' in G.edges(list(G.edges())[0]):
            edge_weights = [G[u][v].get('weight', 1) for u, v in G.edges()]
            max_weight = max(edge_weights)
            edge_widths = [0.5 + (w/max_weight) * 2 for w in edge_weights]
            edge_alphas = [0.2 + (w/max_weight) * 0.4 for w in edge_weights]
        else:
            edge_widths = 0.5
            edge_alphas = 0.3
            
        nx.draw_networkx_edges(G, pos, alpha=edge_alphas, width=edge_widths, 
                              edge_color='#666666')
    
  
    nodes = nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                                  node_size=node_sizes, alpha=0.8, 
                                  cmap=colormap, linewidths=1, 
                                  edgecolors='white')
    
  
    if min_label_degree is None:
        min_label_degree = max(1, np.percentile(list(degrees.values()), 80))
    
    important_nodes = sorted([(node, deg) for node, deg in degrees.items() 
                             if deg >= min_label_degree], 
                            key=lambda x: x[1], reverse=True)[:max_labels]
    
    labels = {}
    for node, _ in important_nodes:
        label = str(node)
        if len(label) > 25:
            words = label.split()
            if len(words) > 3:
                label = ' '.join(words[:3]) + '...'
            else:
                label = label[:22] + '...'
        labels[node] = label
    
  
    label_pos = {}
    for node in labels:
        x, y = pos[node]
        label_pos[node] = (x, y + 0.02)
    
    nx.draw_networkx_labels(G, label_pos, labels, font_size=9, 
                           font_weight='bold', bbox=dict(boxstyle='round,pad=0.2',
                           facecolor='white', alpha=0.8, edgecolor='none'))
    
  
    if nodes:
        cbar = plt.colorbar(nodes, shrink=0.8, pad=0.02)
        if node_attr:
            cbar.set_label('Node Attribute', rotation=270, labelpad=15)
        else:
            cbar.set_label('Node Degree', rotation=270, labelpad=15)
    
    plt.title(title, fontsize=16, pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def create_concept_cooccurrence_network_original(facts_df, links_df, concepts_df, min_cooccurrence=2):
    """Original co-occurrence network but cleaned up"""
    print("\n🔗 ORIGINAL CONCEPT CO-OCCURRENCE NETWORK:")
    print("-" * 60)
    
  
    concept_cooccurrence = defaultdict(int)
    
  
    fact_to_concepts = defaultdict(set)
    for _, link in links_df.iterrows():
        if link['type'] == 'fact_to_concept':
            fact_to_concepts[link['source']].add(link['target'])
    
  
    for fact, concepts in fact_to_concepts.items():
        concepts_list = list(concepts)
        for i, concept1 in enumerate(concepts_list):
            for concept2 in concepts_list[i+1:]:
                pair = tuple(sorted([concept1, concept2]))
                concept_cooccurrence[pair] += 1
    
    print(f"Found {len(concept_cooccurrence)} concept pairs")
    strong_cooccurrences = {pair: count for pair, count in concept_cooccurrence.items() 
                           if count > min_cooccurrence}
    print(f"Strong co-occurrences (>{min_cooccurrence}): {len(strong_cooccurrences)}")
    
  
    print("Most frequently co-occurring concepts:")
    for (c1, c2), freq in sorted(concept_cooccurrence.items(), key=lambda x: x[1], reverse=True)[:10]:
        c1_short = c1[:30] + '...' if len(c1) > 30 else c1
        c2_short = c2[:30] + '...' if len(c2) > 30 else c2
        print(f"  '{c1_short}' + '{c2_short}': {freq} times")
    
  
    G_cooccur = nx.Graph()
    for (c1, c2), weight in concept_cooccurrence.items():
        if weight > min_cooccurrence:
            G_cooccur.add_edge(c1, c2, weight=weight)
    
    if len(G_cooccur.nodes()) == 0:
        print("No co-occurrence network to visualize")
        return G_cooccur
    
    plt.figure(figsize=(16, 12))
    pos = nx.spring_layout(G_cooccur, k=1.5, iterations=50)
    
  
    edges = G_cooccur.edges()
    weights = [G_cooccur[u][v]['weight'] for u, v in edges]
    max_weight = max(weights) if weights else 1
    edge_widths = [w/max_weight * 4 + 0.5 for w in weights]
    nx.draw_networkx_edges(G_cooccur, pos, width=edge_widths, alpha=0.6, edge_color='gray')
    
  
    degrees = dict(G_cooccur.degree())
    node_sizes = [degrees[node] * 50 + 200 for node in G_cooccur.nodes()]
    nx.draw_networkx_nodes(G_cooccur, pos, node_size=node_sizes, 
                          node_color='lightblue', alpha=0.8, linewidths=1, edgecolors='navy')
    
  
    important_nodes = [node for node in G_cooccur.nodes() if G_cooccur.degree(node) > 2]
    labels = {}
    for node in important_nodes:
        label = node[:15] + '...' if len(node) > 15 else node
        labels[node] = label
    
    nx.draw_networkx_labels(G_cooccur, pos, labels, font_size=8, 
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.title('Original Concept Co-occurrence Network\n(concepts that appear together)', fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return G_cooccur

def create_fact_network_original(facts_df, links_df):
    """Original fact-to-fact network visualization but cleaned up"""
    print("\n🌐 ORIGINAL FACT RELATIONSHIP NETWORK:")
    print("-" * 60)
    
    fact_to_fact_links = links_df[links_df['type'] == 'fact_to_fact']
    print(f"Total fact-to-fact relationships: {len(fact_to_fact_links)}")
    
    if len(fact_to_fact_links) == 0:
        print("No fact-to-fact relationships found")
        return nx.DiGraph()
    
  
    G_facts = nx.DiGraph()
    for _, link in fact_to_fact_links.iterrows():
      
        source_short = link['source'][:50] + '...' if len(link['source']) > 50 else link['source']
        target_short = link['target'][:50] + '...' if len(link['target']) > 50 else link['target']
        G_facts.add_edge(source_short, target_short)
    
    print(f"Fact network has {len(G_facts.nodes())} nodes and {len(G_facts.edges())} edges")
    
  
    in_degrees = dict(G_facts.in_degree())
    out_degrees = dict(G_facts.out_degree())
    
    print("\nMost referenced facts (high in-degree):")
    for fact, degree in sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {degree} refs: {fact}")
    
    print("\nFacts that reference many others (high out-degree):")
    for fact, degree in sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {degree} refs: {fact}")
    
  
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G_facts, k=0.8, iterations=50)
    
  
    node_sizes = [in_degrees[node] * 100 + 100 for node in G_facts.nodes()]
    
  
    nx.draw_networkx_edges(G_facts, pos, alpha=0.6, arrows=True, arrowsize=15, 
                          edge_color='gray', width=1)
    
    nodes = nx.draw_networkx_nodes(G_facts, pos, node_size=node_sizes, 
                                  node_color='orange', alpha=0.7, 
                                  linewidths=1, edgecolors='darkorange')
    
  
    high_degree_nodes = [node for node in G_facts.nodes() 
                        if (in_degrees[node] + out_degrees[node]) > 1]
    
    labels = {}
    for node in high_degree_nodes:
      
        label = node[:25] + '...' if len(node) > 25 else node
        labels[node] = label
    
    nx.draw_networkx_labels(G_facts, pos, labels, font_size=8,
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))
    
    plt.title('Original Fact-to-Fact Reference Network\n(node size = how often referenced)', fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return G_facts

def create_centrality_influence_analysis_original(concepts_df, facts_df, links_df):
    """Original centrality and influence analysis but cleaned up"""
    print("\n⭐ ORIGINAL CONCEPT CENTRALITY & INFLUENCE ANALYSIS:")
    print("-" * 60)
    
  
    G = nx.Graph()
    for concept in concepts_df['name']:
        G.add_node(concept)
    
    fact_concept_links = links_df[links_df['type'] == 'fact_to_concept']
    for _, link in fact_concept_links.iterrows():
        if link['target'] in G.nodes:
            G.add_edge(link['source'], link['target'])
    
    if len(G.nodes()) == 0:
        print("No concept network to analyze")
        return
    
  
    degree_centrality = nx.degree_centrality(G)
    
  
    concept_degree = {node: centrality for node, centrality in degree_centrality.items() 
                     if node in concepts_df['name'].values}
    
    if len(concept_degree) == 0:
        print("No concept centralities to analyze")
        return
        
  
    plt.figure(figsize=(16, 12))
    
  
    top_degree = sorted(concept_degree.items(), key=lambda x: x[1], reverse=True)[:15]
    concepts, scores = zip(*top_degree)
    
  
    display_concepts = []
    for c in concepts:
        if len(c) > 30:
            display_concepts.append(c[:27] + '...')
        else:
            display_concepts.append(c)
    
    y_pos = range(len(display_concepts))
    bars = plt.barh(y_pos, scores, color='skyblue', alpha=0.8, edgecolor='navy')
    
    plt.yticks(y_pos, display_concepts)
    plt.xlabel('Degree Centrality')
    plt.title('Original Concept Degree Centrality Analysis\n(Direct Connections in Network)')
    plt.gca().invert_yaxis()
    
  
    for i, (bar, score) in enumerate(zip(bars, scores)):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{score:.3f}', va='center', fontsize=9)
    
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()
    
    print("Top 10 most central concepts:")
    for i, (concept, score) in enumerate(top_degree[:10], 1):
        concept_short = concept[:50] + '...' if len(concept) > 50 else concept
        print(f"  {i:2d}. {concept_short}: {score:.3f}")

def plot_network_evolution_original(concepts_df, facts_df):
    """Original network evolution analysis but cleaned up"""
    print("\n📈 ORIGINAL NETWORK EVOLUTION ANALYSIS:")
    print("-" * 60)
    
    if 'generation' not in concepts_df.columns:
        print("No generation data available")
        return
    
  
    gen_concepts = defaultdict(list)
    for _, concept in concepts_df.iterrows():
        gen_concepts[concept['generation']].append(concept['name'])
    
  
    gen_counts = concepts_df['generation'].value_counts().sort_index()
    
    plt.figure(figsize=(15, 10))
    
  
    plt.subplot(2, 1, 1)
    plt.plot(gen_counts.index, gen_counts.values, 'o-', linewidth=3, markersize=8, 
             color='steelblue', markerfacecolor='lightblue', markeredgecolor='steelblue')
    plt.title('Original: Concept Creation Over Generations', fontsize=14)
    plt.xlabel('Generation')
    plt.ylabel('New Concepts')
    plt.grid(True, alpha=0.3)
    plt.gca().set_facecolor('#f8f9fa')
    
  
    plt.subplot(2, 1, 2)
    cumulative = gen_counts.cumsum()
    plt.plot(cumulative.index, cumulative.values, 's-', color='darkgreen', linewidth=3, 
             markersize=8, markerfacecolor='lightgreen', markeredgecolor='darkgreen')
    plt.title('Original: Cumulative Concepts Over Time', fontsize=14)
    plt.xlabel('Generation')
    plt.ylabel('Total Concepts')
    plt.grid(True, alpha=0.3)
    plt.gca().set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    plt.show()
    
  
    max_gen = concepts_df['generation'].max()
    recent_gens = range(max_gen-5, max_gen+1)
    old_gens = range(1, max_gen-5)
    
    recent_concepts = set()
    old_concepts = set()
    
    for gen in recent_gens:
        if gen in gen_concepts:
            recent_concepts.update(gen_concepts[gen])
    for gen in old_gens:
        if gen in gen_concepts:
            old_concepts.update(gen_concepts[gen])
    
    new_concepts = recent_concepts - old_concepts
    persistent_concepts = recent_concepts & old_concepts
    
    print(f"Concepts only in recent generations ({len(new_concepts)}):")
    for concept in list(new_concepts)[:8]:
        concept_short = concept[:60] + '...' if len(concept) > 60 else concept
        print(f"  {concept_short}")
    
    print(f"\nPersistent concepts across time ({len(persistent_concepts)}):")
    for concept in list(persistent_concepts)[:8]:
        concept_short = concept[:60] + '...' if len(concept) > 60 else concept
        print(f"  {concept_short}")

def create_concept_cooccurrence_network(facts_df, links_df, min_cooccurrence=2):
    """New enhanced co-occurrence network"""
    print("\n🔗 ENHANCED CONCEPT CO-OCCURRENCE NETWORK:")
    print("-" * 50)
    
  
    fact_to_concepts = defaultdict(set)
    for _, link in links_df.iterrows():
        if link['type'] == 'fact_to_concept':
            fact_to_concepts[link['source']].add(link['target'])
    
  
    concept_cooccurrence = defaultdict(int)
    for fact, concepts in fact_to_concepts.items():
        concepts_list = list(concepts)
        for i, concept1 in enumerate(concepts_list):
            for concept2 in concepts_list[i+1:]:
                pair = tuple(sorted([concept1, concept2]))
                concept_cooccurrence[pair] += 1
    
  
    strong_cooccurrences = {pair: count for pair, count in concept_cooccurrence.items() 
                           if count >= min_cooccurrence}
    
    print(f"Found {len(concept_cooccurrence)} concept pairs")
    print(f"Strong co-occurrences (≥{min_cooccurrence}): {len(strong_cooccurrences)}")
    
    if not strong_cooccurrences:
        print("No strong co-occurrences found")
        return nx.Graph()
    
  
    G = nx.Graph()
    for (c1, c2), weight in strong_cooccurrences.items():
        G.add_edge(c1, c2, weight=weight)
    
  
    create_clean_network_viz(G, 
                           title="Enhanced Concept Co-occurrence Network\n(concepts that appear together in facts)",
                           figsize=(20, 16),
                           max_labels=20)
    
  
    print("\nMost frequent co-occurrences:")
    sorted_cooccur = sorted(strong_cooccurrences.items(), key=lambda x: x[1], reverse=True)[:10]
    for (c1, c2), count in sorted_cooccur:
        c1_short = c1[:35] + '...' if len(c1) > 35 else c1
        c2_short = c2[:35] + '...' if len(c2) > 35 else c2
        print(f"  {count:2d}× '{c1_short}' + '{c2_short}'")
    
    return G

def create_fact_reference_network(facts_df, links_df):
    """New enhanced fact reference network"""
    print("\n📄 ENHANCED FACT REFERENCE NETWORK:")
    print("-" * 50)
    
    fact_links = links_df[links_df['type'] == 'fact_to_fact']
    
    if len(fact_links) == 0:
        print("No fact-to-fact references found")
        return nx.DiGraph()
    
  
    G = nx.DiGraph()
    for _, link in fact_links.iterrows():
        G.add_edge(link['source'], link['target'])
    
    print(f"Network: {G.number_of_nodes()} facts, {G.number_of_edges()} references")
    
    if G.number_of_nodes() == 0:
        return G
    
  
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    
    print(f"\nMost referenced facts:")
    top_referenced = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)[:5]
    for fact, refs in top_referenced:
        fact_short = fact[:70] + '...' if len(fact) > 70 else fact
        print(f"  {refs}× {fact_short}")
    
  
    node_attrs = {node: in_degrees[node] for node in G.nodes()}
    
  
    create_clean_network_viz(G, 
                           title="Enhanced Fact Reference Network\n(arrows show which facts reference others)",
                           node_attr=node_attrs,
                           figsize=(18, 14),
                           max_labels=12)
    
    return G

def discover_kg_patterns():
    """Main analysis function with both original and enhanced visualizations"""
    
    db_path = os.path.expanduser("~/npcsh_history.db")
    conn = sqlite3.connect(db_path)
    
    concepts_df = pd.read_sql_query("SELECT * FROM kg_concepts", conn)
    facts_df = pd.read_sql_query("SELECT * FROM kg_facts", conn)
    links_df = pd.read_sql_query("SELECT * FROM kg_links", conn)
    
    print("=" * 80)
    print("🧠 COMPLETE KNOWLEDGE GRAPH NETWORK ANALYSIS")
    print("=" * 80)
    
  
    create_concept_cooccurrence_network_original(facts_df, links_df, concepts_df)
    create_fact_network_original(facts_df, links_df)
    create_centrality_influence_analysis_original(concepts_df, facts_df, links_df)
    plot_network_evolution_original(concepts_df, facts_df)
    
  
    create_concept_cooccurrence_network(facts_df, links_df)
    create_fact_reference_network(facts_df, links_df)
    
  
    print(f"\n📊 SUMMARY STATISTICS:")
    print("-" * 30)
    print(f"Total concepts: {len(concepts_df):,}")
    print(f"Total facts: {len(facts_df):,}")
    print(f"Total links: {len(links_df):,}")
    
    conn.close()

if __name__ == "__main__":
    discover_kg_patterns()