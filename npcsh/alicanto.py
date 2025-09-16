import json
import requests
import argparse
import os
import subprocess
import tempfile
import random 
import shutil
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict, field
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from npcpy.tools import auto_tools
from npcpy.llm_funcs import get_llm_response
from npcpy.data.web import search_web
from npcpy.npc_compiler import NPC, Team
from npcsh._state import NPCSH_CHAT_MODEL, NPCSH_CHAT_PROVIDER

from litellm.exceptions import Timeout, ContextWindowExceededError
import pandas as pd
import numpy as np

from npcsh.wander import perform_single_wandering

@dataclass
class ResearchStep:
    step: int
    thought: str
    action: str
    outcome: str

@dataclass
class SubAgentTrace:
    hypothesis: str
    agent_name: str
    agent_persona: str
    steps: List[ResearchStep] = field(default_factory=list)
    final_files: Dict[str, str] = field(default_factory=dict)
    was_successful: bool = False

@dataclass
class Paper:
    title: str = ""
    abstract: str = ""
    introduction: List[str] = field(default_factory=list)
    methods: List[str] = field(default_factory=list)
    results: List[str] = field(default_factory=list)
    discussion: List[str] = field(default_factory=list)

def create_file(filename: str, content: str) -> str:
    filepath = os.path.abspath(filename)
    if os.path.exists(filepath):
        return f"Error: File '{filename}' already exists. Use append_to_file or replace_in_file to modify."
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        f.write(content)
    return f"File '{filename}' created successfully."

def append_to_file(filename: str, content: str) -> str:
    filepath = os.path.abspath(filename)
    if not os.path.exists(filepath):
        return f"Error: File '{filename}' not found. Use create_file first."
    with open(filepath, 'a') as f:
        f.write("\n" + content)
    return f"Content appended to '{filename}'."

def replace_in_file(filename: str, old_content: str, new_content: str) -> str:
    filepath = os.path.abspath(filename)
    if not os.path.exists(filepath):
        return f"Error: File '{filename}' not found."
    with open(filepath, 'r') as f:
        file_contents = f.read()
    file_contents = file_contents.replace(old_content, new_content)
    with open(filepath, 'w') as f:
        f.write(file_contents)
    return f"Content in '{filename}' replaced."

def read_file(filename: str) -> str:
    filepath = os.path.abspath(filename)
    if not os.path.exists(filepath):
        return f"Error: File '{filename}' not found."
    with open(filepath, 'r') as f:
        return f.read()

def list_files(directory: str = ".") -> List[str]:
    return os.listdir(directory)



from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATASET_CACHE = None
SEARCH_INDEX = None

def load_and_combine_datasets() -> pd.DataFrame:
    all_papers = []
    
    try:
        research_papers = load_dataset("ta-datalab/research_papers", split="train")
        for paper in research_papers:
            all_papers.append({
                'title': paper.get('title', ''),
                'abstract': paper.get('abstract', ''),
                'authors': paper.get('authors', []),
                'year': paper.get('year', None),
                'venue': paper.get('venue', ''),
                'url': paper.get('url', ''),
                'paperId': paper.get('id', ''),
                'citationCount': 0,
                'source': 'research_papers'
            })
    except Exception as e:
        print(f"Failed to load ta-datalab/research_papers: {e}")
    
    try:
        ml_papers = load_dataset("CShorten/ML-ArXiv-Papers", split="train")
        for paper in ml_papers:
            all_papers.append({
                'title': paper.get('title', ''),
                'abstract': paper.get('abstract', ''),
                'authors': paper.get('authors', '').split(', ') if paper.get('authors') else [],
                'year': paper.get('year', None),
                'venue': 'arXiv',
                'url': paper.get('url', ''),
                'paperId': paper.get('id', ''),
                'citationCount': 0,
                'source': 'ml_arxiv'
            })
    except Exception as e:
        print(f"Failed to load CShorten/ML-ArXiv-Papers: {e}")
    

    df = pd.DataFrame(all_papers)
    df = df.dropna(subset=['title', 'abstract'])
    df = df[df['abstract'].str.len() > 50]
    return df

def create_search_index(df: pd.DataFrame):
    search_texts = df['title'].fillna('') + ' ' + df['abstract'].fillna('')
    vectorizer = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(search_texts)
    return {'vectorizer': vectorizer, 'tfidf_matrix': tfidf_matrix, 'dataframe': df}

def initialize_dataset_search():
    global DATASET_CACHE, SEARCH_INDEX
    if DATASET_CACHE is None:
        DATASET_CACHE = load_and_combine_datasets()
    if SEARCH_INDEX is None:
        SEARCH_INDEX = create_search_index(DATASET_CACHE)
    return SEARCH_INDEX

import time

LAST_S2_REQUEST_TIME = 0
S2_RATE_LIMIT_DELAY = 30

def search_semantic_scholar(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    global LAST_S2_REQUEST_TIME
    
    api_key = os.environ.get('S2_API_KEY')
    if not api_key:
        return []
    
    current_time = time.time()
    time_since_last = current_time - LAST_S2_REQUEST_TIME
    
    if time_since_last < S2_RATE_LIMIT_DELAY:
        sleep_time = S2_RATE_LIMIT_DELAY - time_since_last
        print(f"Rate limiting: still need {sleep_time:.2f}s before S2 request")
        return None
    
    LAST_S2_REQUEST_TIME = time.time()
    
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    headers = {"x-api-key": api_key}
    params = {
        "query": query,
        "limit": limit,
        "fields": "title,abstract,authors,year,citationCount,url,tldr"
    }
    print('Semantic SCholar calls')
    try:
        response = requests.get(url, headers=headers, params=params, 
                              timeout=30)
        print('semantic scholar response')
        response.raise_for_status()
        return response.json().get('data', [])
    except requests.exceptions.RequestException as e:
        print(f"Semantic Scholar API error: {e}")
        return []

def search_papers(query: str, limit: int = 10) -> List[Dict]:
    s2_results = search_semantic_scholar(query, limit)
    if s2_results:
        return s2_results
    
    search_index = initialize_dataset_search()
    query_vector = search_index['vectorizer'].transform([query])
    similarities = cosine_similarity(query_vector, search_index['tfidf_matrix']).flatten()
    top_indices = similarities.argsort()[-limit:][::-1]
    results = [search_index['dataframe'].iloc[idx].to_dict() for idx in top_indices if similarities[idx] > 0.01]
    return results

def execute_shell_command(command: str) -> Dict[str, Any]:
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=60)
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    except Exception as e:
        return {"success": False, "stderr": str(e)}

def update_paper(paper_state: Paper, section: str, content: str) -> Paper:
    if not hasattr(paper_state, section):
        return paper_state
    target_section = getattr(paper_state, section)
    if isinstance(target_section, list):
        target_section.append(content)
    else:
        setattr(paper_state, section, content)
    return paper_state

def get_creative_ideas_for_stuck_agent(
    problem_description: str,
    npc: NPC,
    model: str,
    provider: str
) -> str:
    print(f"\n--- SUB-AGENT {npc.name} IS STUCK, INITIATING WANDER ---")
    _, _, raw_brainstorm, _, _ = perform_single_wandering(
        problem=problem_description,
        npc=npc,
        model=model,
        provider=provider
    )
    return raw_brainstorm


@dataclass
class FileProvenance:
    filename: str
    step_history: List[Tuple[int, str, str, str]] = field(default_factory=list)  

def get_filesystem_state() -> Dict[str, str]:
    import hashlib
    files = {}
    for f in os.listdir("."):
        if os.path.isfile(f):
            with open(f, 'rb') as file:
                content = file.read()
                files[f] = hashlib.md5(content).hexdigest()[:8]
    return files

def summarize_step(thought: str, 
                   action: str, 
                   outcome: str, 
                   fs_before: Dict[str, str], 
                   fs_after: Dict[str, str], 
                   file_provenance: Dict[str, FileProvenance], 
                   step_num: int, 
                   model: str, 
                   provider: str, 
                   npc: NPC) -> str:
    
    import hashlib
    import os
    
    
    current_files = {}
    for f in os.listdir("."):
        if os.path.isfile(f):
            with open(f, 'rb') as file:
                content = file.read()
                current_files[f] = {
                    'size': len(content),
                    'checksum': hashlib.md5(content).hexdigest()[:8]
                }
    
    
    for f in fs_after:
        if f not in file_provenance:
            file_provenance[f] = FileProvenance(filename=f)
        
        change_summary = ""
        if f not in fs_before:
            change_summary = f"Created with {current_files[f]['size']} bytes"
            file_provenance[f].step_history.append((step_num, "CREATED", fs_after[f], change_summary))
        elif fs_before.get(f) != fs_after[f]:
            change_summary = f"Modified to {current_files[f]['size']} bytes"
            file_provenance[f].step_history.append((step_num, "MODIFIED", fs_after[f], change_summary))
    
    
    provenance_summary = []
    for filename, prov in file_provenance.items():
        history = "; ".join([f"Step {step}: {action} ({checksum}) - {changes}" for step, action, checksum, changes in prov.step_history])
        provenance_summary.append(f"{filename}: {history}")
    
    prompt = f"""AGENT'S REASONING: {thought}

            AGENT'S ACTION: {action}  
        AGENT'S CLAIMED OUTCOME: {outcome}

        COMPLETE FILE PROVENANCE:
        {chr(10).join(provenance_summary)}

        CURRENT FILESYSTEM:
        Files: {list(current_files.keys())}
        Details: {current_files}

Explain plainly what happened and whether the actions produced any measurable effects. If the agent thinks then it is likely time to direct it to 
carry out a specific action. 

Return JSON with "summary" and "next_step" keys.""" + """

{
    "summary": " a summary of what they did and claimed and the extent to which it produced the intended outcome .", 
    "next_step": "The concrete next step for the agent to carry out in their research.

}
"""
    
    response = get_llm_response(prompt, model=model, provider=provider, npc=npc, format='json')
    summary_data = response.get('response')

    return summary_data



from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import csv
import os
from datetime import datetime

Base = declarative_base()

class AlicantoPersona(Base):
    __tablename__ = 'alicanto_personas'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255))
    birth_year = Column(Integer)
    location = Column(Text)
    leader = Column(Text)
    interests = Column(Text)
    worldview = Column(Text)
    approach = Column(Text)
    persona_text = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

def save_persona_to_databases(persona_data: dict):
    """Save persona to both SQLite and CSV for persistence"""
    
    
    db_path = os.path.expanduser("~/npcsh_history.db")
    engine = create_engine(f'sqlite:///{db_path}')
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    
    persona = AlicantoPersona(
        name=persona_data.get('name'),
        birth_year=persona_data.get('birth_year'),
        location=persona_data.get('location'),
        leader=persona_data.get('leader'),
        interests=json.dumps(persona_data.get('interests', [])),
        worldview=persona_data.get('worldview'),
        approach=persona_data.get('approach'),
        persona_text=persona_data.get('persona_text')
    )
    
    session.add(persona)
    session.commit()
    session.close()
    
    
    csv_dir = os.path.expanduser("~/.npcsh/npc_team")
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, "alicanto_personas.csv")
    
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as csvfile:
        fieldnames = ['name', 'birth_year', 'location', 'leader', 'interests', 
                     'worldview', 'approach', 'persona_text', 'created_at']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
            
        writer.writerow({
            **persona_data,
            'interests': json.dumps(persona_data.get('interests', [])),
            'created_at': datetime.now().isoformat()
        })

def generate_sub_agent_personas(topic: str, num_agents: int, model: str, provider: str, npc: NPC) -> List[Dict[str, str]]:
    personas = []
    for i in range(num_agents):
        birth_year = random.randint(-32665, 32665)
        teen_year = birth_year + 16
        
        json_template = """
{
  "name": "culturally appropriate full name for someone born in """ + str(birth_year) + """",
  "location": "specific city/region where they were born in """ + str(birth_year) + """",
  "leader": "who ruled their region when they were 16 years old in """ + str(teen_year) + """",
  "interests": ["3-5 specific interests/obsessions they had as a teenager in """ + str(teen_year) + """"],
  "worldview": "one sentence describing their fundamental perspective shaped by growing up in that era",
  "approach": "how their historical background influences their way of thinking"
}
"""
        
        prompt = f"Generate a unique persona for someone born in {birth_year}. Return JSON:\n{json_template}\n\nMake this person feel real and historically grounded. Consider: technological context, cultural movements, economic conditions, wars, discoveries happening in {teen_year}."
        

        response = get_llm_response(
            prompt,
            model=model,
            provider=provider,
            npc=npc,
            format='json'
        )
        
        new_persona = response.get('response')
        if isinstance(new_persona, str):
            new_persona = json.loads(new_persona)
        
        persona_text = f"You are {new_persona.get('name')}, born {birth_year} in {new_persona.get('location')}, came of age under {new_persona.get('leader')}. Your interests were: {', '.join(new_persona.get('interests', []))}. {new_persona.get('worldview')} {new_persona.get('approach')}"
        
        
        persona_data = {
            'name': new_persona.get('name'),
            'birth_year': birth_year,
            'location': new_persona.get('location'),
            'leader': new_persona.get('leader'),
            'interests': new_persona.get('interests', []),
            'worldview': new_persona.get('worldview'),
            'approach': new_persona.get('approach'),
            'persona_text': persona_text
        }
        
        
        save_persona_to_databases(persona_data)
        
        personas.append({
            "name": new_persona.get('name'),
            "persona": persona_text
        })
    
    return personas
    

def create_sub_agent(
    model: str, 
    provider: str, 
    hypothesis: str, 
    name: str, 
    persona: str
) -> NPC:
    
    def wander_wrapper(problem_description: str) -> str:
        return get_creative_ideas_for_stuck_agent(
            problem_description, 
            agent, 
            model, 
            provider
        )







    tools = [
        create_file, 
        append_to_file, 
        replace_in_file, 
        read_file, 
        list_files, 
        execute_shell_command, 
        search_papers, 
        wander_wrapper, 
        search_web 
    ]
        
    agent = NPC(
        name=name,
        model=model,
        provider=provider,
        primary_directive=persona,
        tools=tools
    )
    
    return agent



def sub_agent_trace(hypothesis: str, 
                    persona: Dict[str, str], 
                    user_query: str, 
                    model: str, 
                    provider: str, 
                    max_steps: int = 50) -> SubAgentTrace:
    agent_name = persona.get("name")
    agent_persona = persona.get("persona")
    agent = create_sub_agent(model, provider, hypothesis, agent_name, agent_persona)
    
    trace = SubAgentTrace(hypothesis=hypothesis, agent_name=agent_name, agent_persona=agent_persona)
    summarized_history = []
    file_provenance = {}
    created_files = set()
    summary = {}
    
    major_step = 0
    
    while major_step < max_steps:
        fs_before = get_filesystem_state()
        
        provenance_summary = []
        for filename, prov in file_provenance.items():
            history = "; ".join([f"Step {step}: {action} ({checksum}) - {changes}" for step, action, checksum, changes in prov.step_history])
            provenance_summary.append(f"{filename}: {history}")
        
        history_str = "\n".join(summarized_history)
        next_step_text = f"This is the next step suggested by your advisor. : BEGIN NEXT_STEP: {summary.get('next_step')} END NEXT STEP" if summary else ""
        
        initial_prompt = f"""
Test the following hypothesis: '{hypothesis}' as related to the user query: '{user_query}'. 
Only focus on your specific hypothesis, other agents are being tasked with other aspects of the problem.

Use bash commands to carry out research through the execute_shell_command.
Adjust files with `replace_in_file` and use `read_file` and `list_files` to verify file states and file creation.
Create files with create_file()

Test with execute_shell_command when needed
Get unstuck with wander_wrapper

When you have a definitive result, say RESEARCH_COMPLETE.

FILE PROVENANCE HISTORY:
{chr(10).join(provenance_summary)}

CURRENT FILES: {list(fs_before.keys())}

COMPLETE ACTION HISTORY:
BEGIN HISTORY
`
{history_str}
`
END HISTORy

What specific action will you take next to test your hypothesis?
AVAILABLE TOOLS: create_file, append_to_file, replace_in_file, read_file, list_files, execute_shell_command, wander_wrapper, search_web .

Do not repeat actions. Do not constantly think unless you need to brainstorm or wander. Use `execute_shell_command` for anything complicated beyond a simple file read, replace, create.
Use `search_web` with provider of {os.environ.get('NPCSH_SEARCH_PROVIDER') } to look up items if you are struggling to understand why errors are happening with code execution.
Do not waste time re-verifying the same package versins or libraries when you can explicitly look up usage patterns that are up to date. Do not assume that your generated code will be correct the first time or up to date
amd if you are finding irreconcilable errors that you cannot seem to figure out locally then you need to search. For example, if you assume a python package you installed like `sqlite-vector' is importable like
"from sqlite.vector" and keep running into import or module errors, it it probably because you need to look up the correct way to access the library. It may have been that you would need to import "sqlite_vector" or "sql_vector".
There is no way to know this information a priori and instead of wasting time verifying pip installations, its better to look for actual usage patterns, either by inspecting the source code of the pip package itself or simply by
searching the web.

This should guide your next steps:

`{next_step_text} `

Your goal is to research. To set up experiments, create figures that can be included in a latex document report, and  produce data outputs as well in csvs for verification and reusability and reproducibility.


Do not use seaborn. On matplotlib plots, do not use grids or titles. 
"""
        
        print(f"\n{'='*80}")
        print(f"AUTONOMOUS LOOP {major_step + 1} FOR {agent_name}")
        print(f"{'='*80}")
        print(f"HYPOTHESIS: {hypothesis}")
        print(f"FILES BEFORE: {list(fs_before.keys())}")
        
        messages = []
        all_thoughts = []
        all_actions = []
        all_outcomes = []
        
        for micro_step in range(11):
            print(f"\n--- Micro-step {micro_step + 1}/4 ---")
            
            if micro_step == 0:
                current_prompt = initial_prompt
                print("SENDING INITIAL RESEARCH PROMPT")
            else:
                current_prompt = "Continue your work. What's your next action?"
                print(f"SENDING CONTINUATION PROMPT: '{current_prompt}'")
            try:
                response = agent.get_llm_response(current_prompt, 
                                                messages=messages, 
                                                auto_process_tool_calls=True)
            except Timeout:
                continue
            except ContextWindowExceededError:
                break
            messages = response.get('messages', [])
            
            thought = response.get('response') 
            if thought is None:
                thought = ''
                print("WARNING: No thought received from agent")
            else:
                print(f"AGENT THOUGHT: {thought[:200]}{'...' if len(thought) > 200 else ''}")
                all_thoughts.append(thought)
            
            if thought and "RESEARCH_COMPLETE" in thought.upper():
                print(f"✓ RESEARCH COMPLETED at micro-step {micro_step + 1}")
                break
            
            if response.get('tool_results'):
                tool_results = response['tool_results']
                print(f"TOOLS USED: {len(tool_results)} tool(s)")
                
                for i, res in enumerate(tool_results):
                    tool_name = res.get('tool_name')
                    args = res.get('arguments', {})
                    result = res.get('result')
                    
                    print(f"  Tool {i+1}: {tool_name}({args})")
                    for arg, item in args.items():
                        print(f"    {arg}: {item}")
                    if isinstance(result, str) and len(result) > 150:
                        print(f"    Result: {result[:150]}...")
                    else:
                        print(f"    Result: {result}")
                
                action_str = ", ".join([f"{res['tool_name']}({res.get('arguments', {})})" for res in tool_results])
                outcomes = []
                
                for res in tool_results:
                    if res['tool_name'] in ['create_file', 'append_to_file', 'replace_in_file']:
                        filename = res.get('arguments', {}).get('filename')
                        if filename:
                            created_files.add(filename)
                            if os.path.exists(filename):
                                trace.was_successful = True
                                print(f"  ✓ File created: {filename}")
                    
                    result_data = res.get('result')
                    outcomes.append(str(result_data))
                
                outcome_str = " | ".join(outcomes)
                all_actions.append(action_str)
                all_outcomes.append(outcome_str)
            else:
                print("NO TOOLS USED - Agent only provided reasoning")
        
        fs_after = get_filesystem_state()
        print(f"\nFILES AFTER: {list(fs_after.keys())}")
        
        new_files = set(fs_after.keys()) - set(fs_before.keys())
        if new_files:
            print(f"NEW FILES CREATED: {list(new_files)}")
        
        combined_thought = " ".join(all_thoughts)
        combined_action = " | ".join(filter(None, all_actions))
        combined_outcome = " | ".join(filter(None, all_outcomes))
        
        print(f"\nCOMPRESSING AUTONOMOUS SESSION...")
        print(f"THOUGHTS: {len(all_thoughts)} messages")
        print(f"ACTIONS: {len(all_actions)} tool uses")
        
        summary = summarize_step(combined_thought, 
                                 combined_action,
                                 combined_outcome,
                                 fs_before, 
                                 fs_after, 
                                 file_provenance, 
                                 major_step + 1, 
                                 model, 
                                 provider, 
                                 agent)
        
        print(f"SUMMARY: {summary.get('summary', 'No summary')}")
        print(f"NEXT STEP: {summary.get('next_step', 'No next step')}")
        
        summarized_history.append(f"Step {major_step + 1}: {summary.get('summary')} ")
        
        trace.steps.append(ResearchStep(
            step=major_step + 1,
            thought=combined_thought,
            action=combined_action,
            outcome=combined_outcome
        ))
        
        if combined_thought and "RESEARCH_COMPLETE" in combined_thought.upper():
            print(f"✓ RESEARCH COMPLETED FOR {agent_name}")
            break
            
        major_step += 1
    
    for filename in created_files:
        if os.path.exists(filename):
            trace.final_files[filename] = read_file(filename)
    
    print(f"\nFINAL RESULTS FOR {agent_name}:")
    print(f"SUCCESS: {trace.was_successful}")
    print(f"FILES CREATED: {list(trace.final_files.keys())}")
    
    return trace




def save_trace_for_training(
        

    traces: List[SubAgentTrace],
    output_dir: str = "./alicanto_traces"
):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"trace_{timestamp}.csv"
    filepath = os.path.join(output_dir, filename)
    
    flattened_data = []
    for trace in traces:
        for step in trace.steps:
            flattened_data.append({
                "hypothesis": trace.hypothesis,
                "agent_name": trace.agent_name,
                "agent_persona": trace.agent_persona,
                "was_successful": trace.was_successful,
                "step": step.step,
                "thought": step.thought,
                "action": step.action,
                "outcome": step.outcome,
                "final_files": json.dumps(trace.final_files)
            })
            
    if not flattened_data:
        return

    df = pd.DataFrame(flattened_data)
    df.to_csv(filepath, index=False)
    
    print(f"Full research trace saved to {filepath}")
    return filepath
def compress_traces_for_synthesis(traces: List[SubAgentTrace], model: str, provider: str, npc: NPC) -> str:
    compressed_summaries = []
    
    for trace in traces:
        steps_summary = []
        for step in trace.steps[-3:]:  # Only last 3 steps
            if step.thought:
                thought_short = step.thought[:100] + "..." if len(step.thought) > 100 else step.thought
            else:
                thought_short = "No thought recorded"
            
            if step.action:
                action_short = step.action[:100] + "..." if len(step.action) > 100 else step.action
            else:
                action_short = "No action taken"
                
            steps_summary.append(f"Step {step.step}: {thought_short} | {action_short}")
        
        files_created = list(trace.final_files.keys()) if trace.final_files else []
        
        compressed_summaries.append({
            "agent": trace.agent_name,
            "hypothesis": trace.hypothesis,
            "success": trace.was_successful,
            "key_steps": steps_summary,
            "files_created": files_created,
            "final_file_count": len(files_created)
        })
    
    return json.dumps(compressed_summaries, indent=2)
def format_paper_as_latex(paper: Paper, authors: List[str]) -> str:
    author_string = ", ".join(authors)
    introduction_content = "\n\n".join(paper.introduction)
    methods_content = "\n\n".join(paper.methods)
    results_content = "\n\n".join(paper.results)
    discussion_content = "\n\n".join(paper.discussion)

    return f"""
\\documentclass{{article}}
\\title{{{paper.title}}}
\\author{{{author_string}}}
\\date{{\\today}}
\\begin{{document}}
\\maketitle
\\begin{{abstract}}
{paper.abstract}
\\end{{abstract}}
\\section*{{Introduction}}
{introduction_content}
\\section*{{Methods}}
{methods_content}
\\section*{{Results}}
{results_content}
\\section*{{Discussion}}
{discussion_content}
\\end{{document}}
"""

def alicanto(
    query: str,
    num_agents: int = 3,
    max_steps: int = 10,
    model: str = NPCSH_CHAT_MODEL,
    provider: str = NPCSH_CHAT_PROVIDER,
    skip_research: bool = True,
    **kwargs
) -> None:

    print("=== ALICANTO RESEARCH SYSTEM STARTING ===")
    print(f"Query: {query}")
    
    if skip_research:
        print("SKIPPING RESEARCH - GOING DIRECTLY TO PAPER WRITING")
    else:
        print(f"Agents: {num_agents}, Max steps per agent: {max_steps}")
    
    print(f"Model: {model}, Provider: {provider}")
    
    def wander_wrapper_coordinator(problem_description: str) -> str:
        return get_creative_ideas_for_stuck_agent(
            problem_description, 
            alicanto_coordinator, 
            model, 
            provider
        )
    
    alicanto_coordinator = NPC(
        name="Alicanto",
        model=model,
        provider=provider,
        primary_directive="You are Alicanto the mythical bird. You research topics iteratively by writing to LaTeX files and searching for more information.",
        tools=[
            create_file,
            append_to_file,
            replace_in_file,
            read_file,
            list_files,
            execute_shell_command,
            search_papers,
            search_web,
            wander_wrapper_coordinator
        ]
    )

    messages = []
    summarized_history = []
    file_provenance = {}

    if not skip_research:
        print("\n--- Step 1: Generating hypotheses and personas ---")
        
        one_shot_example_hypotheses = """
"example_input": "Investigate the impact of quantum annealing on protein folding.",
"example_output": {
    "hypotheses": [
        "Implementing a quantum annealer simulation for a small peptide chain will identify lower energy states faster than a classical simulated annealing approach.",
        "The choice of qubit connectivity in the quantum annealer's topology significantly impacts the final folded state's accuracy for proteins with long-range interactions.",
        "Encoding the protein's residue interactions as a QUBO problem is feasible for structures up to 50 amino acids before qubit requirements become prohibitive."
    ]
}
"""
        hypotheses_prompt = f"""Based on the following research topic, generate a list of {num_agents} distinct, specific, and empirically testable hypotheses.

TOPIC: "{query}"

Return a JSON object with a single key "hypotheses" which is a list of strings.

Here is an example of the expected input and output format:
{one_shot_example_hypotheses}

Return ONLY the JSON object.
"""
        
        print("Generating hypotheses...")
        response = get_llm_response(
            hypotheses_prompt,
            model=model,
            provider=provider,
            npc=alicanto_coordinator,
            format='json'
        )
        
        if not response or not response.get('response'):
            print("ERROR: Failed to get hypotheses response")
            return
        
        hypotheses = response.get('response').get('hypotheses')
        if not hypotheses:
            print("ERROR: No hypotheses generated")
            return
        
        print(f"Generated {len(hypotheses)} hypotheses:")
        for i, h in enumerate(hypotheses):
            print(f"  {i+1}. {h}")
        
        print("\nGenerating agent personas...")
        personas = generate_sub_agent_personas(
            query,
            num_agents,
            model,
            provider,
            alicanto_coordinator
        )
        
        if not personas:
            print("ERROR: No personas generated")
            return
        
        print(f"Generated {len(personas)} personas:")
        for i, p in enumerate(personas):
            print(f"  {i+1}. {p.get('name')}: {p.get('persona')}")

        print("\n--- Step 2: Delegating hypotheses to Sub-Agents for serial execution ---")
        
        all_traces = []
        for i, hypo in enumerate(hypotheses):
            persona = personas[i % len(personas)]
            print(f"\nStarting sub-agent {i+1}/{len(hypotheses)}")
            trace = sub_agent_trace(
                hypo,
                persona,
                query,
                model,
                provider,
                max_steps
            )
            all_traces.append(trace)
            print(f"Sub-agent {i+1} completed. Success: {trace.was_successful}")

        print(f"\nAll sub-agents completed. Saving traces...")
        save_trace_for_training(all_traces)
        compressed_research = compress_traces_for_synthesis(all_traces, model, provider, alicanto_coordinator)

        print("\n--- Step 3: Creating initial paper structure ---")
        
        author_list = [trace.agent_name for trace in all_traces]
        author_string = ", ".join(author_list)
        
        initial_latex = f"""\\documentclass{{article}}
\\title{{% TODO: TITLE}}
\\author{{{author_string}}}
\\date{{\\today}}
\\begin{{document}}
\\maketitle

\\begin{{abstract}}
% TODO: ABSTRACT
\\end{{abstract}}

\\section{{Introduction}}
% TODO: INTRODUCTION

\\section{{Methods}}
% TODO: METHODS

\\section{{Results}}
% TODO: RESULTS

\\section{{Discussion}}
% TODO: DISCUSSION

\\end{{document}}"""

        create_file("paper.tex", initial_latex)
    else:
        print("\n--- Skipping research phase - loading existing data ---")
        
        if os.path.exists("paper.tex"):
            print("Found existing paper.tex")
        else:
            print("No existing paper.tex found, creating basic template...")
            basic_latex = f"""\\documentclass{{article}}
\\title{{{query.title()}}}
\\author{{Research Team}}
\\date{{\\today}}
\\begin{{document}}
\\maketitle

\\begin{{abstract}}
% TODO: ABSTRACT
\\end{{abstract}}

\\section{{Introduction}}
% TODO: INTRODUCTION

\\section{{Methods}}
% TODO: METHODS

\\section{{Results}}
% TODO: RESULTS

\\section{{Discussion}}
% TODO: DISCUSSION

\\end{{document}}"""
            create_file("paper.tex", basic_latex)
        
        compressed_research = f"Research topic: {query}. Previous research data should be available in local files."

    print("\n--- Step 4: Iterative paper writing ---")
    
    for section_round in range(25):
        print(f"\n--- Section Round {section_round + 1} ---")
        
        fs_before = get_filesystem_state()
        
        provenance_summary = []
        for filename, prov in file_provenance.items():
            history = "; ".join([f"Step {step}: {action} ({checksum}) - {changes}" for step, action, checksum, changes in prov.step_history])
            provenance_summary.append(f"{filename}: {history}")
        
        history_str = "\n".join(summarized_history)
        current_paper = read_file("paper.tex")
        
        initial_prompt = f"""You are writing a research paper about: "{query}" located at ./paper.tex

Research data from sub-agents: {compressed_research}

Current paper content:
{current_paper}

FILE PROVENANCE HISTORY:
{chr(10).join(provenance_summary)}

COMPLETE ACTION HISTORY:
BEGIN HISTORY
{history_str}
END HISTORY

Ensure the paper contains the following sections and that they have a coherent narrative by the end of your work.
work iteratively, so do not worry about making it all in one step.

SECTIONS: Title, Abstract, Intro, Methods, Results, Discussion, Conclusions,

You may choose to add subsections as you wish, but do not do so for the introduction. 

You must ensure citations are properly included in your results and cited with the \cite{{author_year}} format , keeping in mind
to also start and maintain a .bib file separate from any currently provided. be sure to reference this as well. 

Your title short be short, informative, and eye-catching. 
Every section and paragraph should be written in a formal academic style, motivating pieces of information and ensuring
each sentence must flow well into the last, and the paper must have a strong motivation with substantial literature review to establish
the need for the investigation. The paper should focus only on 1-2 major findings, with 5-10 minor findings detailed in the conclusions.
The discussion should primarily focus on commenting on how previous work may be re-interpreted in light of your findings. Do not simply splatter text
into a discussion but be thoughtful and helpful. The discussion should connect to broader works and discuss specifics of those works. Do not simply regurgitate the

Use replace_in_file to update the paper. Use search_papers or search_web if you need more information.

Write 2-4 paragraphs of substantial academic content. Include figures and tables based on the results of the experiments.

Available tools: replace_in_file, read_file, search_papers, search_web, list_files"""

        all_thoughts = []
        all_actions = []
        all_outcomes = []

        for micro_step in range(5):
            print(f"\n--- Micro-step {micro_step + 1}/5  ---")
            
            if micro_step == 0:
                current_prompt = initial_prompt
            else:
                current_prompt = f"continue "
            
            try:
                response = alicanto_coordinator.get_llm_response(
                    current_prompt, 
                    messages=messages, 
                    auto_process_tool_calls=True
                )
                print('response: ', response['response'])
                print('tool calls: ', response['tool_calls'])
                print('tool results: ', response['tool_results'])
                
                messages = response.get('messages', [])
                
                thought = response.get('response') or ""  # Handle None case
                all_thoughts.append(thought)
                
                if response.get('tool_results'):
                    tool_results = response['tool_results']
                    action_str = ", ".join([f"{res['tool_name']}({res.get('arguments', {})})" for res in tool_results])
                    outcomes = [str(res.get('result', '')) for res in tool_results]
                    outcome_str = " | ".join(outcomes)
                    all_actions.append(action_str)
                    all_outcomes.append(outcome_str)
                
            except (Timeout, ContextWindowExceededError):
                break
            except Exception as e:
                print(f"Error in micro-step: {e}")
                break
        
        fs_after = get_filesystem_state()
        
        combined_thought = " ".join(filter(None, all_thoughts))  # Filter out None values
        combined_action = " | ".join(filter(None, all_actions))
        combined_outcome = " | ".join(filter(None, all_outcomes))
        
        print(f"\nCOMPRESSING WRITING SESSION...")
        print(f"THOUGHTS: {len(all_thoughts)} messages")
        print(f"ACTIONS: {len(all_actions)} tool uses")
        
        summary = summarize_step(combined_thought, 
                                 combined_action,
                                 combined_outcome,
                                 fs_before, 
                                 fs_after, 
                                 file_provenance, 
                                 section_round + 1, 
                                 model, 
                                 provider, 
                                 alicanto_coordinator)
        
        print(f"SUMMARY: {summary.get('summary', 'No summary')}")
        print(f"NEXT STEP: {summary.get('next_step', 'No next step')}")
        
        summarized_history.append(f"Round {section_round + 1}: {summary.get('summary')} ")
            
    final_paper = read_file("paper.tex")
    print(f"\n{'='*60}")
    print("FINAL RESEARCH PAPER (LATEX)")
    print("="*60)
    print(final_paper)
    print(f"\nPaper saved as paper.tex")
    
    
    
def main():
    parser = argparse.ArgumentParser(description="Alicanto Multi-Agent Research System")
    parser.add_argument("topic", help="Research topic to investigate")
    parser.add_argument("--num-agents", type=int, default=3, help="Number of sub-agents to run.")
    parser.add_argument("--max-steps", type=int, default=10, help="Maximum steps for each sub-agent.")
    parser.add_argument("--model", default=NPCSH_CHAT_MODEL, help="LLM model to use")
    parser.add_argument("--provider", default=NPCSH_CHAT_PROVIDER, help="LLM provider to use")
    parser.add_argument("--skip-research", action="store_true", help="Skip research phase and go directly to paper writing")
    
    args = parser.parse_args()
    
    alicanto(
        query=args.topic,
        num_agents=args.num_agents,
        max_steps=args.max_steps,
        model=args.model,
        provider=args.provider,
        skip_research=args.skip_research
    )