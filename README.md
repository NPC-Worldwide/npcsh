<p align="center">
  <a href= "https://github.com/npc-worldwide/npcsh/blob/main/docs/npcsh.md"> 
  <img src="https://raw.githubusercontent.com/npc-worldwide/npcsh/main/npcsh/npcsh.png" alt="npcsh logo" width=250></a>
</p> 

# NPC Shell

The NPC shell is a suite of executable command-line programs that allow users to easily interact with NPCs and LLMs through a command line shell. 

Programs within the NPC shell use the properties defined in `~/.npcshrc`, which is generated upon installation and running of `npcsh` for the first time.

To get started:
```
pip install 'npcsh[local]'
```
Once installed, the following CLI tools will be available: `npcsh`, `guac`, `npc` cli, `yap` `pti`, `wander`, and `spool`. 


# npcsh
- An AI-powered shell that parses bash, natural language, and special macro calls, `npcsh` processes your input accordingly, agentically, and automatically.


  - Get help with a task: 
      ```bash
      npcsh:🤖sibiji:gemini-2.5-flash>can you help me identify what process is listening on port 5337? 
      ```
      <p align="center"> 
        <img src="https://raw.githubusercontent.com/npc-worldwide/npcsh/main/test_data/port5337.png" alt="example of running npcsh to check what processes are listening on port 5337", width=600>
      </p>

  - Edit files
      ```bash
      npcsh>please read through the markdown files in the docs folder and suggest changes based on the current implementation in the src folder
      ```

  - **Ask a Generic Question**
    ```bash
    npcsh> has there ever been a better pasta shape than bucatini?
    ```
    
    ```
    .Loaded .env file...                                                                                                                                               
    Initializing database schema...                                                                                                                                                            
    Database schema initialization complete.                                                                                                                                                   
    Processing prompt: 'has there ever been a better pasta shape than bucatini?' with NPC: 'sibiji'...                                                                                         
    • Action chosen: answer_question                                                                                                                                                           
    • Explanation given: The question is a general opinion-based inquiry about pasta shapes and can be answered without external data or jinx invocation.                                      
    ...............................................................................                                                                                                            
    Bucatini is certainly a favorite for many due to its unique hollow center, which holds sauces beautifully. Whether it's "better" is subjective and depends on the dish and personal        
    preference. Shapes like orecchiette, rigatoni, or trofie excel in different recipes. Bucatini stands out for its versatility and texture, making it a top contender among pasta shapes!    
    ```


  - **Search the Web**
    ```bash
    /search "cal golden bears football schedule" -sp perplexity
    ```
    <p align="center">
        <img src="https://raw.githubusercontent.com/npc-worldwide/npcsh/main/test_data/search_example.png" alt="example of search results", width=600>
     </p>

  - **Computer Use**
    ```bash
    /plonk 'find out the latest news on cnn'
    ```

  - **Generate Image**
    ```bash
    /vixynt 'generate an image of a rabbit eating ham in the brink of dawn' model='gpt-image-1' provider='openai'
    ```
      <p align="center">
        <img src="https://raw.githubusercontent.com/npc-worldwide/npcsh/main/test_data/rabbit.PNG" alt="a rabbit eating ham in the bring of dawn", width=250>
      </p>
  - **Generate Video**
    ```bash
    /roll 'generate a video of a hat riding a dog'
    ```
    <!--
      <p align="center">
        <img src="https://raw.githubusercontent.com/npc-worldwide/npcsh/main/test_data/hat_video.mp4" alt="video of a hat riding a dog", width=250>
      </p> -->

  - **Serve an NPC Team**
    ```bash
    /serve --port 5337 --cors='http://localhost:5137/'
    ```
  - **Screenshot Analysis**
    ```bash
    /ots
    ```


# Macros
-  activated by invoking `/<command> ...` in `npcsh`, macros can be called in bash or through the `npc` CLI. In our examples, we provide both `npcsh` calls as well as bash calls with the `npc` cli where relevant. For converting any `/<command>` in `npcsh` to a bash version, replace the `/` with `npc ` and the macro command will be invoked as a positional argument. Some, like breathe, flush,
    
    - `/alicanto` - Conduct deep research with multiple perspectives, identifying gold insights and cliff warnings
    - `/brainblast` - Execute an advanced chunked search on command history
    - `/breathe` - Condense context on a regular cadence
    - `/compile` - Compile NPC profiles
    - `/corca` - Enter the Corca MCP-powered agentic shell. Usage: /corca [--mcp-server-path path]                  
    - `/flush` - Flush the last N messages
    - `/guac` - Enter guac mode
    - `/help` - Show help for commands, NPCs, or Jinxs. Usage: /help 
    - `/init` - Initialize NPC project
    - `/jinxs` - Show available jinxs for the current NPC/Team
    - `/npc-studio` - Start npc studio                                                                                                                           
    - `/ots` - Take screenshot and analyze with vision model
    - `/plan` - Execute a plan command
    - `/plonk` - Use vision model to interact with GUI. Usage: /plonk <task description>
    - `/pti` - Use pardon-the-interruption mode to interact with reasoning model LLM
    - `/rag` - Execute a RAG command using ChromaDB embeddings with optional file input (-f/--file)
    - `/roll` - generate a video with video generation model
    - `/sample` - Send a prompt directly to the LLM
    - `/search` - Execute a web search command
    - `/serve` - Serve an NPC Team server.
    - `/set` - Set configuration values
    - `/sleep` - Evolve knowledge graph with options for dreaming. 
    - `/spool` - Enter interactive chat (spool) mode with an npc with fresh context or files for rag
    - `/trigger` - Execute a trigger command
    - `/vixynt` - Generate and edit images from text descriptions using local models, openai, gemini
    - `/wander` - A method for LLMs to think on a problem by switching between states of high temperature and low temperature
    - `/yap` - Enter voice chat (yap) mode
    
    ## Common Command-Line Flags:
    
    ```
    Flag              Shorthand    | Flag              Shorthand    | Flag              Shorthand    | Flag              Shorthand   
    ------------------------------ | ------------------------------ | ------------------------------ | ------------------------------
    --attachments     (-a)         | --height          (-h)         | --num_npcs        (-num_n)     | --team            (-tea)      
    --config_dir      (-con)       | --igmodel         (-igm)       | --output_file     (-o)         | --temperature     (-t)      
    --cors            (-cor)       | --igprovider      (-igp)       | --plots_dir       (-pl)        | --top_k                       
    --creativity      (-cr)        | --lang            (-l)         | --port            (-po)        | --top_p                       
    --depth           (-d)         | --max_tokens      (-ma)        | --provider        (-pr)        | --vmodel          (-vm)       
    --emodel          (-em)        | --messages        (-me)        | --refresh_period  (-re)        | --vprovider       (-vp)       
    --eprovider       (-ep)        | --model           (-mo)        | --rmodel          (-rm)        | --width           (-w)        
    --exploration     (-ex)        | --npc             (-np)        | --rprovider       (-rp)        |                               
    --format          (-f)         | --num_frames      (-num_f)     | --sprovider       (-s)         |                               
    ```
    '

## Read the Docs
To see more about how to use the macros and modes in the NPC Shell, read the docs at [npc-shell.readthedocs.io](https://npc-shell.readthedocs.io/en/latest/)


## Inference Capabilities
- `npcsh` works with local and enterprise LLM providers through its LiteLLM integration, allowing users to run inference from Ollama, LMStudio, vLLM, MLX, OpenAI, Anthropic, Gemini, and Deepseek, making it a versatile tool for both simple commands and sophisticated AI-driven tasks. 

## NPC Studio
There is a graphical user interface that makes use of the NPC Toolkit through the NPC Studio. See the open source code for NPC Studio [here](https://github.com/npc-worldwide/npc-studio). Download the executables at [our website](https://enpisi.com/npc-studio).

## Mailing List
Interested to stay in the loop and to hear the latest and greatest about `npcpy`, `npcsh`, and NPC Studio? Be sure to sign up for the [newsletter](https://forms.gle/n1NzQmwjsV4xv1B2A)!

## Support
If you appreciate the work here, [consider supporting NPC Worldwide with a monthly donation](https://buymeacoffee.com/npcworldwide), [buying NPC-WW themed merch](https://enpisi.com/shop), or hiring us to help you explore how to use the NPC Toolkit and AI tools to help your business or research team, please reach out to info@npcworldwi.de .


## Installation
`npcsh` is available on PyPI and can be installed using pip. Before installing, make sure you have the necessary dependencies installed on your system. Below are the instructions for installing such dependencies on Linux, Mac, and Windows. If you find any other dependencies that are needed, please let us know so we can update the installation instructions to be more accommodating.

### Linux install
<details>  <summary> Toggle </summary>
  
```bash

# these are for audio primarily, skip if you dont need tts
sudo apt-get install espeak
sudo apt-get install portaudio19-dev python3-pyaudio
sudo apt-get install alsa-base alsa-utils
sudo apt-get install libcairo2-dev
sudo apt-get install libgirepository1.0-dev
sudo apt-get install ffmpeg

# for triggers
sudo apt install inotify-tools


#And if you don't have ollama installed, use this:
curl -fsSL https://ollama.com/install.sh | sh

ollama pull llama3.2
ollama pull llava:7b
ollama pull nomic-embed-text
pip install npcsh
# if you want to install with the API libraries
pip install 'npcsh[lite]'
# if you want the full local package set up (ollama, diffusers, transformers, cuda etc.)
pip install 'npcsh[local]'
# if you want to use tts/stt
pip install 'npcsh[yap]'
# if you want everything:
pip install 'npcsh[all]'

```

</details>


### Mac install

<details>  <summary> Toggle </summary>

```bash
#mainly for audio
brew install portaudio
brew install ffmpeg
brew install pygobject3

# for triggers
brew install inotify-tools


brew install ollama
brew services start ollama
ollama pull llama3.2
ollama pull llava:7b
ollama pull nomic-embed-text
pip install npcsh
# if you want to install with the API libraries
pip install npcsh[lite]
# if you want the full local package set up (ollama, diffusers, transformers, cuda etc.)
pip install npcsh[local]
# if you want to use tts/stt
pip install npcsh[yap]

# if you want everything:
pip install npcsh[all]
```
</details>

### Windows Install

<details>  <summary> Toggle </summary>
Download and install ollama exe.

Then, in a powershell. Download and install ffmpeg.

```powershell
ollama pull llama3.2
ollama pull llava:7b
ollama pull nomic-embed-text
pip install npcsh
# if you want to install with the API libraries
pip install 'npcsh[lite]'
# if you want the full local package set up (ollama, diffusers, transformers, cuda etc.)
pip install 'npcsh[local]'
# if you want to use tts/stt
pip install 'npcsh[yap]'

# if you want everything:
pip install 'npcsh[all]'
```
As of now, npcsh appears to work well with some of the core functionalities like /ots and /yap.

</details>

### Fedora Install (under construction)

<details>  <summary> Toggle </summary>
  
```bash
python3-dev #(fixes hnswlib issues with chroma db)
xhost +  (pyautogui)
python-tkinter (pyautogui)
```

</details>

## Startup Configuration and Project Structure
After `npcsh` has been pip installed, `npcsh`, `guac`, `pti`, `spool`, `yap` and the `npc` CLI can be used as command line tools. To initialize these correctly, first start by starting the NPC shell:
```bash
npcsh
```
When initialized, `npcsh` will generate a .npcshrc file in your home directory that stores your npcsh settings.
Here is an example of what the .npcshrc file might look like after this has been run.
```bash
# NPCSH Configuration File
export NPCSH_INITIALIZED=1
export NPCSH_CHAT_PROVIDER='ollama'
export NPCSH_CHAT_MODEL='llama3.2'
export NPCSH_DB_PATH='~/npcsh_history.db'
```

`npcsh` also comes with a set of jinxs and NPCs that are used in processing. It will generate a folder at ~/.npcsh/ that contains the tools and NPCs that are used in the shell and these will be used in the absence of other project-specific ones. Additionally, `npcsh` records interactions and compiled information about npcs within a local SQLite database at the path specified in the .npcshrc file. This will default to ~/npcsh_history.db if not specified. When the data mode is used to load or analyze data in CSVs or PDFs, these data will be stored in the same database for future reference.

The installer will automatically add this file to your shell config, but if it does not do so successfully for whatever reason you can add the following to your .bashrc or .zshrc:

```bash
# Source NPCSH configuration
if [ -f ~/.npcshrc ]; then
    . ~/.npcshrc
fi
```

We support inference via all providers supported by litellm. For openai-compatible providers that are not explicitly named in litellm, use simply `openai-like` as the provider. The default provider must be one of `['openai','anthropic','ollama', 'gemini', 'deepseek', 'openai-like']` and the model must be one available from those providers.

To use tools that require API keys, create an `.env` file in the folder where you are working or place relevant API keys as env variables in your ~/.npcshrc. If you already have these API keys set in a ~/.bashrc or a ~/.zshrc or similar files, you need not additionally add them to ~/.npcshrc or to an `.env` file. Here is an example of what an `.env` file might look like:

```bash
export OPENAI_API_KEY="your_openai_key"
export ANTHROPIC_API_KEY="your_anthropic_key"
export DEEPSEEK_API_KEY='your_deepseek_key'
export GEMINI_API_KEY='your_gemini_key'
export PERPLEXITY_API_KEY='your_perplexity_key'
```


 Individual npcs can also be set to use different models and providers by setting the `model` and `provider` keys in the npc files.
 Once initialized and set up, you will find the following in your ~/.npcsh directory:
```bash
~/.npcsh/
├── npc_team/           # Global NPCs
│   ├── jinxs/          # Global tools
│   └── assembly_lines/ # Workflow pipelines
│   └── example.npc  # globally available npc 
│   └── global.ctx  # global context file



```
For cases where you wish to set up a project specific set of NPCs, jinxs, and assembly lines, add a `npc_team` directory to your project and `npcsh` should be able to pick up on its presence, like so:
```bash
./npc_team/            # Project-specific NPCs
├── jinxs/             # Project jinxs #example jinx next
│   └── example.jinx
└── assembly_lines/    # Agentic Workflows
    └── example.line
└── models/    # Project workflows
    └── example.model
└── example1.npc        # Example NPC
└── example2.npc        # Example NPC
└── team.ctx            # Example ctx


```

## Contributing
Contributions are welcome! Please submit issues and pull requests on the GitHub repository.


## License
This project is licensed under the MIT License.
