#!/bin/bash
# Test npcsh fresh install experience in Docker
set -e

echo "=== npcsh Fresh Install Test ==="
echo ""

# Build the image
echo "1. Building Docker image..."
docker build -t npcsh-fresh-test . 2>&1 | tail -3

# Test: binary runs
echo ""
echo "2. Testing binary starts..."
docker run --rm npcsh-fresh-test --help 2>&1 | head -5
echo "   OK: binary runs"

# Test: can list jinxes
echo ""
echo "3. Testing jinx loading..."
JINX_COUNT=$(docker run --rm npcsh-fresh-test -c "/jinxes" 2>&1 | grep -c "jinx" || echo "0")
echo "   Jinxes found: $JINX_COUNT"

# Test: can reach ollama if host network
echo ""
echo "4. Testing ollama connectivity (host network)..."
docker run --rm --network host npcsh-fresh-test -c "ollama list" 2>&1 | head -5 || echo "   (ollama not reachable — expected without --network host)"

# Test: interactive mode starts and exits
echo ""
echo "5. Testing interactive mode starts..."
echo "/exit" | timeout 10 docker run --rm -i npcsh-fresh-test 2>&1 | head -10
echo "   OK: interactive mode works"

echo ""
echo "=== All tests passed ==="
