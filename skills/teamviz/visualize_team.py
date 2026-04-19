import os
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict

team_path = context.get('team_path') or ''
save_path = context.get('save') or ''

# Find team path
if not team_path:
    if os.path.exists('./npc_team'):
        team_path = './npc_team'
    elif os.path.exists(os.path.expanduser('~/.npcsh/npc_team')):
        team_path = os.path.expanduser('~/.npcsh/npc_team')
    else:
        output = "No npc_team found. Specify team_path."
        exit()

team_path = Path(team_path)

# Load NPCs
npcs = {}
for npc_file in team_path.glob("*.npc"):
    try:
        with open(npc_file, 'r') as f:
            data = yaml.safe_load(f)
        npcs[npc_file.stem] = {
            'jinxes': data.get('jinxes', []),
            'directive': data.get('primary_directive', '')[:100],
            'model': data.get('model', ''),
        }
    except Exception as e:
        print(f"Error loading {npc_file}: {e}")

# Load team context
team_name = "NPC Team"
forenpc = None
for ctx_file in team_path.glob("*.ctx"):
    try:
        with open(ctx_file, 'r') as f:
            ctx = yaml.safe_load(f)
        team_name = ctx.get('name', team_name)
        forenpc = ctx.get('forenpc')
    except:
        pass

# Discover all jinxes
jinxes_dir = team_path / "jinxes"
all_jinxes = {}
if jinxes_dir.exists():
    for jinx_file in jinxes_dir.rglob("*.jinx"):
        rel_path = jinx_file.relative_to(jinxes_dir)
        jinx_name = jinx_file.stem
        # Categorize by location
        if 'bin' in str(rel_path):
            category = 'bin'
        elif 'lib' in str(rel_path):
            parts = str(rel_path).split(os.sep)
            category = parts[1] if len(parts) > 1 else 'lib'
        else:
            category = 'other'
        all_jinxes[jinx_name] = {'path': str(rel_path), 'category': category}

# Build graph
G = nx.DiGraph()

# Add team node
G.add_node(team_name, type='team', size=3000)

# Add NPC nodes
for npc_name, npc_data in npcs.items():
    node_type = 'forenpc' if npc_name == forenpc else 'npc'
    G.add_node(npc_name, type=node_type, size=2000)
    G.add_edge(team_name, npc_name, relation='has_npc')

# Track jinx usage
jinx_users = defaultdict(list)
npc_jinx_patterns = {}

for npc_name, npc_data in npcs.items():
    patterns = npc_data.get('jinxes', [])
    npc_jinx_patterns[npc_name] = patterns

    # Resolve patterns to actual jinxes
    for pattern in patterns:
        pattern = str(pattern)
        if '*' in pattern:
            # Glob pattern like lib/browser/*
            base = pattern.replace('/*', '').replace('*', '')
            for jname, jdata in all_jinxes.items():
                if base in jdata['path']:
                    jinx_users[jname].append(npc_name)
        else:
            # Exact match
            jname = pattern.split('/')[-1]
            if jname in all_jinxes:
                jinx_users[jname].append(npc_name)

# Add jinx nodes (only ones that are used)
jinx_categories = defaultdict(list)
for jname, users in jinx_users.items():
    if jname in all_jinxes:
        cat = all_jinxes[jname]['category']
        jinx_categories[cat].append(jname)

        # Size based on usage
        size = 500 + len(users) * 200
        G.add_node(jname, type='jinx', category=cat, size=size, users=len(users))

        for user in users:
            G.add_edge(user, jname, relation='uses')

# Check for delegation relationships
for npc_name, npc_data in npcs.items():
    directive = npc_data.get('directive', '').lower()
    for other_npc in npcs.keys():
        if other_npc != npc_name and other_npc in directive:
            if 'delegate' in directive:
                G.add_edge(npc_name, other_npc, relation='delegates_to')

# Create visualization
fig, ax = plt.subplots(1, 1, figsize=(16, 12))

# Layout
pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

# Color map
color_map = {
    'team': '#FF6B6B',
    'forenpc': '#4ECDC4',
    'npc': '#45B7D1',
    'jinx': '#96CEB4',
}

# Draw by type
for node_type in ['team', 'forenpc', 'npc', 'jinx']:
    nodes = [n for n, d in G.nodes(data=True) if d.get('type') == node_type]
    sizes = [G.nodes[n].get('size', 1000) for n in nodes]
    nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=color_map[node_type],
                            node_size=sizes, alpha=0.9, ax=ax)

# Draw edges by type
delegate_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('relation') == 'delegates_to']
use_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('relation') == 'uses']
has_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('relation') == 'has_npc']

nx.draw_networkx_edges(G, pos, edgelist=has_edges, edge_color='#999', alpha=0.5, ax=ax)
nx.draw_networkx_edges(G, pos, edgelist=use_edges, edge_color='#96CEB4', alpha=0.3, ax=ax)
nx.draw_networkx_edges(G, pos, edgelist=delegate_edges, edge_color='#FF6B6B',
                        width=2, style='dashed', ax=ax, arrows=True, arrowsize=20)

# Labels
nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax)

# Legend
legend_elements = [
    plt.scatter([], [], c=color_map['team'], s=200, label='Team'),
    plt.scatter([], [], c=color_map['forenpc'], s=200, label='ForeNPC'),
    plt.scatter([], [], c=color_map['npc'], s=200, label='NPC'),
    plt.scatter([], [], c=color_map['jinx'], s=200, label='Jinx'),
]
ax.legend(handles=legend_elements, loc='upper left')

ax.set_title(f"{team_name} - Team Structure", fontsize=14, fontweight='bold')
ax.axis('off')

plt.tight_layout()

if save_path:
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    output = f"Saved to {save_path}"
else:
    plt.show()
    output = "Displayed team visualization"

# Text summary
summary = "\n\n--- Team Summary ---\n"
summary += f"Team: {team_name}\n"
summary += f"ForeNPC: {forenpc or 'None'}\n"
summary += f"NPCs ({len(npcs)}): {', '.join(sorted(npcs.keys()))}\n"
summary += f"Total Jinxes: {len(all_jinxes)}\n"
summary += "\nJinxes by category:\n"
for cat, jlist in sorted(jinx_categories.items()):
    summary += f"  {cat}: {len(jlist)} jinxes\n"

summary += "\nMost shared jinxes:\n"
shared = sorted(jinx_users.items(), key=lambda x: len(x[1]), reverse=True)[:10]
for jname, users in shared:
    if len(users) > 1:
        summary += f"  {jname} (used by {len(users)}): {', '.join(users)}\n"

output = output + summary
