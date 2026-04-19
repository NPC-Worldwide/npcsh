import os
import datetime
import traceback
from npcpy.llm_funcs import get_facts
from npcpy.memory.command_history import CommandHistory, format_memory_context
from sqlalchemy import text

limit_str = context.get('limit', '50')
user_context = context.get('context', '')
llm_model = context.get('model')
llm_provider = context.get('provider')
current_npc = context.get('npc')
current_team = context.get('team')

# Resolve model/provider
if not llm_model and current_npc and current_npc.model:
    llm_model = current_npc.model
if not llm_provider and current_npc and current_npc.provider:
    llm_provider = current_npc.provider
if not llm_model: llm_model = state.chat_model if state else "llama3.2"
if not llm_provider: llm_provider = state.chat_provider if state else "ollama"

npc_name = current_npc.name if current_npc else "default"
team_name = current_team.name if current_team else "default"
current_path = os.getcwd()

try:
    limit = int(limit_str)
except ValueError:
    limit = 50

command_history = None
try:
    db_path = os.getenv("NPCSH_DB_PATH", os.path.expanduser("~/npcsh_history.db"))
    command_history = CommandHistory(db_path)
    engine = command_history.engine

    # Get IDs of messages that already have memories extracted
    with engine.connect() as conn:
        existing_ids = set()
        try:
            rows = conn.execute(text("SELECT DISTINCT message_id FROM memory_lifecycle"))
            for row in rows:
                if row[0]:
                    # message_ids are like "conv_id_HHMMSS_idx" — extract conv prefix
                    existing_ids.add(row[0])
        except:
            pass

    # Get recent assistant messages to extract from
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT conversation_id, content, timestamp
            FROM conversation_history
            WHERE role = 'assistant'
            ORDER BY timestamp DESC
            LIMIT :lim
        """), {"lim": limit * 2})  # fetch extra to filter
        messages = []
        for row in rows:
            messages.append({
                'conversation_id': row[0],
                'content': row[1],
                'timestamp': row[2]
            })

    # Build conversation chunks (group by conversation_id)
    from collections import defaultdict
    convos = defaultdict(list)
    for msg in messages:
        convos[msg['conversation_id']].append(msg['content'])

    # Get memory examples for context
    memory_examples_dict = command_history.get_memory_examples_for_context(
        npc=npc_name,
        team=team_name,
        directory_path=current_path
    )
    memory_context = format_memory_context(memory_examples_dict)

    # Extraction guide — tells the model what good memories look like
    full_context = """Extract memories as precise factual statements written in clear academic prose. Each memory should read like a sentence from a research paper or technical report.

    Focus on: technical decisions and rationale, user preferences and workflow patterns, project architecture and constraints, problems encountered and their resolutions, tools/models/configurations chosen, domain knowledge demonstrated, relationships between systems and components.

    Do NOT extract: trivial UI interactions, generic assistant capabilities, pleasantries, anything obvious from reading the code.

    Example memories from a conversation about building a GIS mapping pane:
    - "The application follows a pattern where tool panes are opened via toolbar buttons and registered as content types in the pane renderer map, as demonstrated by Scherzo (audio), Vixynt (images), and Cartoglyph (mapping)."
    - "OSINT geographic data is fetched from Overpass API for structured OSM tag queries and Nominatim for geocoding, with results cached per viewport and auto-refreshed on map pan."
    - "The decision to use Leaflet with react-leaflet over alternatives was driven by compatibility with React 18 and the need for CSP-compliant tile loading in an Electron renderer."

    Example memories from a conversation about fixing Windows compatibility:
    - "The GGUF model source detection logic used forward-slash path matching that failed on Windows due to backslash path separators, requiring normalization before string comparison."
    - "LM Studio stores downloaded models at ~/.cache/lm-studio/models on all platforms including Windows, contrary to the initial assumption that Windows would use APPDATA."
    - "Settings persistence failures on Windows were caused by the Flask backend not receiving the NPCSH_BASE environment variable, resulting in silent write failures to incorrect paths."

    Example memories from a conversation about authentication architecture:
    - "All authentication, billing, and subscription management is handled exclusively through Clerk with no Stripe integration, established as a core architectural constraint."
    - "Content Security Policy headers must whitelist Clerk-owned domains for the OAuth flow to complete within the Electron BrowserWindow."

    Each memory must stand alone as a meaningful, falsifiable claim. No two memories should make substantially similar claims."""
    if memory_context:
        full_context = full_context + "\n\nPreviously extracted memories for reference:\n" + memory_context
    if user_context:
        full_context = full_context + "\n\n" + user_context

    total_extracted = 0
    conversations_processed = 0

    for conv_id, contents in list(convos.items())[:limit]:
        conversation_text = "\n".join(contents[:10])  # cap per conversation
        if len(conversation_text.strip()) < 50:
            continue

        try:
            facts = get_facts(
                conversation_text,
                model=llm_model,
                provider=llm_provider,
                npc=current_npc,
                context=full_context
            )
        except Exception as e:
            print(f"Error extracting from conversation {conv_id}: {e}")
            continue

        if facts:
            for i, fact in enumerate(facts):
                ts = datetime.datetime.now().strftime('%H%M%S')
                msg_id = f"{conv_id}_{ts}_{i}"
                command_history.add_memory_to_database(
                    message_id=msg_id,
                    conversation_id=conv_id,
                    npc=npc_name,
                    team=team_name,
                    directory_path=current_path,
                    initial_memory=fact.get('statement', str(fact)),
                    status="pending_approval",
                    model=llm_model,
                    provider=llm_provider,
                    final_memory=None
                )
                total_extracted += 1
            conversations_processed += 1

    context['output'] = (
        f"Memory extraction complete.\n"
        f"- Conversations processed: {conversations_processed}\n"
        f"- Memories extracted: {total_extracted} (pending approval)"
    )

except Exception as e:
    traceback.print_exc()
    context['output'] = f"Error during memory extraction: {e}"
finally:
    if command_history:
        command_history.close()
