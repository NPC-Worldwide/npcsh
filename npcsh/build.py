import shutil
import textwrap
from pathlib import Path


def build_flask_server(config, **kwargs):
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    server_script = output_dir / 'npc_server.py'
    
    server_code = textwrap.dedent(f'''
      from npcpy.serve import start_flask_server
    from npcpy.npc_compiler import Team
    from sqlalchemy import create_engine
    
    if __name__ == "__main__":
        team_path = os.path.join(
            os.path.dirname(__file__), 
            "npc_team"
        )
        db_path = os.path.expanduser("~/npcsh_history.db")
        
        db_conn = create_engine(f'sqlite:///{{db_path}}')
        team = Team(team_path=team_path, db_conn=db_conn)
        
        start_flask_server(
            port={config['port']},
            cors_origins={config.get('cors_origins')},
            teams={{"main": team}},
            npcs=team.npcs,
            db_path=db_path,
            user_npc_directory=os.path.expanduser(
                "~/.npcsh/npc_team"
            )
        )
    ''')
    
    server_script.write_text(server_code)
    
    shutil.copytree(
        config['team_path'], 
        output_dir / 'npc_team',
        dirs_exist_ok=True
    )
    
    requirements = output_dir / 'requirements.txt'
    requirements.write_text(
        'npcsh\n'
        'flask\n'
        'flask-cors\n'
        'sqlalchemy\n'
    )
    
    readme = output_dir / 'README.md'
    readme.write_text(textwrap.dedent(f'''
    # NPC Team Server
    
    Run: python npc_server.py
    
    Server will be available at http://localhost:{config['port']}
    
    For pyinstaller standalone:
    pyinstaller --onefile npc_server.py
    '''))
    
    return {
        "output": f"Flask server built in {output_dir}", 
        "messages": kwargs.get('messages', [])
    }


def build_docker_compose(config, **kwargs):
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    shutil.copytree(
        config['team_path'], 
        output_dir / 'npc_team',
        dirs_exist_ok=True
    )
    
    dockerfile = output_dir / 'Dockerfile'
    dockerfile.write_text(textwrap.dedent('''
    FROM python:3.11-slim
    
    WORKDIR /app
    
    COPY requirements.txt .
    RUN pip install --no-cache-dir -r requirements.txt
    
    COPY npc_team ./npc_team
    COPY npc_server.py .
    
    EXPOSE 5337
    
    CMD ["python", "npc_server.py"]
    '''))
    
    compose = output_dir / 'docker-compose.yml'
    compose.write_text(textwrap.dedent(f'''
    version: '3.8'
    
    services:
      npc-server:
        build: .
        ports:
          - "{config['port']}:{config['port']}"
        volumes:
          - npc-data:/root/.npcsh
        environment:
          - NPCSH_DB_PATH=/root/npcsh_history.db
    
    volumes:
      npc-data:
    '''))
    
    build_flask_server(config, **kwargs)
    
    return {
        "output": f"Docker compose built in {output_dir}. Run: docker-compose up", 
        "messages": kwargs.get('messages', [])
    }


def build_cli_executable(config, **kwargs):
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cli_script = output_dir / 'npc_cli.py'
    
    cli_code = textwrap.dedent('''
    import sys
    from npcsh._state import setup_shell, execute_command, initial_state
    from npcsh.routes import router
    
    def main():
        if len(sys.argv) < 2:
            print("Usage: npc_cli <command>")
            sys.exit(1)
        
        command = " ".join(sys.argv[1:])
        
        command_history, team, npc = setup_shell()
        initial_state.npc = npc
        initial_state.team = team
        
        state, result = execute_command(
            command, 
            initial_state, 
            router=router
        )
        
        output = result.get('output') if isinstance(result, dict) else result
        print(output)
    
    if __name__ == "__main__":
        main()
    ''')
    
    cli_script.write_text(cli_code)
    
    shutil.copytree(
        config['team_path'], 
        output_dir / 'npc_team',
        dirs_exist_ok=True
    )
    
    spec_file = output_dir / 'npc_cli.spec'
    spec_file.write_text(textwrap.dedent('''
    a = Analysis(
        ['npc_cli.py'],
        pathex=[],
        binaries=[],
        datas=[('npc_team', 'npc_team')],
        hiddenimports=[],
        hookspath=[],
        hooksconfig={},
        runtime_hooks=[],
        excludes=[],
        win_no_prefer_redirects=False,
        win_private_assemblies=False,
        cipher=None,
        noarchive=False,
    )
    pyz = PYZ(a.pure, a.zipped_data, cipher=None)
    
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.zipfiles,
        a.datas,
        [],
        name='npc',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        upx_exclude=[],
        runtime_tmpdir=None,
        console=True,
    )
    '''))
    
    return {
        "output": (
            f"CLI executable built in {output_dir}. "
            f"Run: pyinstaller npc_cli.spec"
        ), 
        "messages": kwargs.get('messages', [])
    }


def build_static_site(config, **kwargs):
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    html = output_dir / 'index.html'
    html.write_text(textwrap.dedent(f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>NPC Team Interface</title>
        <style>
            body {{ 
                font-family: monospace; 
                max-width: 800px; 
                margin: 50px auto; 
            }}
            #output {{ 
                white-space: pre-wrap; 
                background: #f5f5f5; 
                padding: 20px; 
                min-height: 300px; 
            }}
        </style>
    </head>
    <body>
        <h1>NPC Team</h1>
        <input id="input" type="text" 
               placeholder="Enter command..." 
               style="width: 100%; padding: 10px;">
        <div id="output"></div>
        
        <script>
        const API_URL = '{config.get("api_url", "http://localhost:5337")}';
        
        document.getElementById('input').addEventListener('keypress', 
            async (e) => {{
            if (e.key === 'Enter') {{
                const cmd = e.target.value;
                e.target.value = '';
                
                const resp = await fetch(`${{API_URL}}/api/stream`, {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify({{
                        commandstr: cmd,
                        conversationId: 'web-session',
                        model: 'llama3.2',
                        provider: 'ollama'
                    }})
                }});
                
                const reader = resp.body.getReader();
                const decoder = new TextDecoder();
                
                while (true) {{
                    const {{done, value}} = await reader.read();
                    if (done) break;
                    
                    const text = decoder.decode(value);
                    document.getElementById('output').textContent += text;
                }}
            }}
        }});
        </script>
    </body>
    </html>
    '''))
    
    return {
        "output": (
            f"Static site built in {output_dir}. "
            f"Serve with: python -m http.server 8000"
        ), 
        "messages": kwargs.get('messages', [])
    }