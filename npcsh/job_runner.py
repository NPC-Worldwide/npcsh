#!/usr/bin/env python3
"""
npcsh Job Runner Daemon

Executes .nsh scripts (npc-shell scripts) as background jobs.
.nsh files contain a sequence of npcsh commands that can:
- Call jinxes (/jinx_name)
- Run shell commands
- Create and persist variables
"""

import os
import sys
import json
import sqlite3
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

from npcsh._state import (
    ShellState,
    execute_command,
    setup_shell,
    initialize_router_with_jinxes,
    initial_state
)
from npcsh.routes import router


class JobState:
    """Minimal state for job execution with variable persistence."""

    def __init__(self, db_path: str, job_name: str):
        self.db_path = db_path
        self.job_name = job_name
        self.variables: Dict[str, Any] = {}
        self.messages = []
        self._load_state()

    def _load_state(self):
        """Load persisted state from database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT state_data FROM job_states
                WHERE job_name = ?
            """, (self.job_name,))
            row = cursor.fetchone()
            if row:
                data = json.loads(row[0])
                self.variables = data.get('variables', {})
                self.messages = data.get('messages', [])
            conn.close()
        except Exception:
            pass

    def save_state(self):
        """Persist state to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS job_states (
                    job_name TEXT PRIMARY KEY,
                    state_data TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                INSERT OR REPLACE INTO job_states (job_name, state_data)
                VALUES (?, ?)
            """, (self.job_name, json.dumps({
                'variables': self.variables,
                'messages': self.messages
            })))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Warning: Could not save state: {e}")


def run_nsh_script(script_path: str, db_path: str, team_dir: str) -> Dict[str, Any]:
    """
    Execute a .nsh script file.

    Args:
        script_path: Path to the .nsh file
        db_path: Path to the npcsh history database
        team_dir: Path to the npc team directory

    Returns:
        Dict with execution results
    """
    script_name = Path(script_path).stem

    # Read the script
    with open(script_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip() and not line.startswith('#')]

    # Initialize minimal shell state
    command_history, team, forenpc = setup_shell()
    state = initial_state
    state.team = team
    state.npc = forenpc
    state.command_history = command_history
    state.current_path = os.getcwd()

    # Initialize router with jinxes
    initialize_router_with_jinxes(team, router)

    # Load persisted job state
    job_state = JobState(db_path, script_name)
    state.variables.update(job_state.variables)
    if job_state.messages:
        state.messages = job_state.messages

    def _substitute_vars(text: str) -> str:
        """Replace $var and ${var} with values from state.variables."""
        import re
        def repl(m):
            var = m.group(1) or m.group(2)
            val = state.variables.get(var, '')
            return str(val) if val is not None else ''
        return re.sub(r'\$\{(\w+)\}|\$(\w+)', repl, text)

    last_output = ""
    results = []
    output_log = []

    for i, line in enumerate(lines):
        # Variable assignment: $var = value
        var_assign = None
        assign_match = __import__('re').match(r'^\s*\$(\w+)\s*=\s*(.+)$', line)
        if assign_match:
            var_assign = assign_match.group(1)
            line = assign_match.group(2).strip()

        # Substitute variables in the line
        substituted = _substitute_vars(line)

        # Handle $_ substitution (last result)
        substituted = substituted.replace('$_', str(last_output))

        print(f"[{script_name}] ({i+1}/{len(lines)}): {substituted[:60]}...")

        try:
            # Strip leading ! for bash commands (job runner runs in agent-like mode)
            if substituted.startswith('!'):
                cmd_to_exec = substituted[1:].strip()
            else:
                cmd_to_exec = substituted

            # Execute the command
            state, output = execute_command(
                cmd_to_exec,
                state,
                review=False,
                router=router,
                command_history=command_history
            )

            # Extract output string
            if isinstance(output, dict):
                out_str = output.get('output', '') or output.get('response', '')
            else:
                out_str = str(output) if output else ''

            # Capture variable assignment
            if var_assign is not None:
                state.variables[var_assign] = out_str
                print(f"  → ${var_assign} = {out_str[:80]}...")

            last_output = out_str

            output_entry = {
                'line': line,
                'line_number': i + 1,
                'success': True,
                'output': out_str,
                'error': None
            }
            results.append(output_entry)
            output_log.append(f"$ {line}")
            output_log.append(out_str)

            # Persist any new variables
            job_state.variables.update(state.variables)
            job_state.messages = state.messages

        except Exception as e:
            error_msg = f"Error executing line {i+1}: {str(e)}"
            print(f"[{script_name}] {error_msg}")
            results.append({
                'line': line,
                'line_number': i + 1,
                'success': False,
                'output': '',
                'error': str(e)
            })
            output_log.append(f"$ {line}")
            output_log.append(f"ERROR: {error_msg}")

    # Save state
    job_state.save_state()

    # Write to log file
    log_dir = Path.home() / '.npcsh' / 'npc_team' / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{script_name}.log"

    with open(log_path, 'a') as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"Job: {script_name}\n")
        f.write(f"Timestamp: {__import__('datetime').datetime.now().isoformat()}\n")
        f.write(f"{'='*60}\n")
        f.write('\n'.join(output_log))
        f.write('\n')

    return {
        'success': all(r['success'] for r in results),
        'results': results,
        'log_path': str(log_path),
        'lines_executed': len(results)
    }


def main():
    parser = argparse.ArgumentParser(description='npcsh Job Runner')
    parser.add_argument('script', help='Path to .nsh script file')
    parser.add_argument('--db', default='~/npcsh_history.db', help='Database path')
    parser.add_argument('--team', default='~/.npcsh/npc_team', help='Team directory')
    parser.add_argument('--daemon', action='store_true', help='Run as daemon (watch mode)')
    parser.add_argument('--interval', type=int, default=60, help='Watch interval in seconds')

    args = parser.parse_args()

    db_path = os.path.expanduser(args.db)
    team_dir = os.path.expanduser(args.team)
    script_path = os.path.expanduser(args.script)

    if not os.path.exists(script_path):
        print(f"Error: Script not found: {script_path}")
        sys.exit(1)

    if args.daemon:
        # Watch mode - run script periodically
        import time
        print(f"Running in daemon mode (interval: {args.interval}s)")
        while True:
            try:
                result = run_nsh_script(script_path, db_path, team_dir)
                print(f"Execution complete: {result['lines_executed']} lines")
                if not result['success']:
                    print("Warning: Some commands failed")
            except Exception as e:
                print(f"Execution failed: {e}")
            time.sleep(args.interval)
    else:
        # Single execution
        result = run_nsh_script(script_path, db_path, team_dir)
        print(f"\nExecution complete: {result['lines_executed']} lines")
        print(f"Log: {result['log_path']}")
        sys.exit(0 if result['success'] else 1)


if __name__ == '__main__':
    main()
