import pandas as pd

query = ({{ sql_query | tojson }}).strip()
if not query:
    context['output'] = "Usage: /sql <query>"
else:
    try:
        df = pd.read_sql_query(query, npc.db_conn)
        context['output'] = df.to_string()
    except Exception as e:
        context['output'] = "SQL Error: " + str(e)
