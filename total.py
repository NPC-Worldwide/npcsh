import json
with open(/Users/caug/npcww/npc-core/npcsh/inventory.json, r) as f:
    data = json.load(f)
total = sum(item[quantity] * item[price] for item in data)
print(total)
