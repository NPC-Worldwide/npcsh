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
      ```
      npcsh:🤖sibiji:gemini-2.5-flash>can you help me identify what process is listening on port 5337? 
      ```
      <p align="center"> 
        <img src="https://raw.githubusercontent.com/npc-worldwide/npcsh/main/test_data/port5337.png" alt="example of running npcsh to check what processes are listening on port 5337", width=600>
      </p>
    - Edit files
    ```
      npcsh:🤖sibiji:gemini-2.5-flash>can you help me identify what process is listening on port 5337? 
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

    - ## TL; DR:
    - `/alicanto` - Conduct deep research with multiple perspectives, identifying gold insights and cliff warnings
    - `/brainblast` - Execute an advanced chunked search on command history
    - `/breathe` - Condense context on a regular cadence
    - `/compile` - Compile NPC profiles
    - `/flush` - Flush the last N messages
    - `/guac` - Enter guac mode
    - `/help` - Show help for commands, NPCs, or Jinxs. Usage: /help 
    - `/init` - Initialize NPC project
    - `/jinxs` - Show available jinxs for the current NPC/Team
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

    - ## `/alicanto`: a research exploration agent flow. 

      <p align="center"><a href ="https://github.com/npc-worldwide/npcsh/blob/main/docs/alicanto.md"> 
        <img src="https://raw.githubusercontent.com/npc-worldwide/npcsh/main/npcsh/npc_team/alicanto.png" alt="logo for deep research", width=250></a>
      </p>

    - Examples:
      ```bash
      # npcsh
      /alicanto "What are the implications of quantum computing for cybersecurity?"
      /alicanto "How might climate change impact global food security?" --num-npcs 8 --depth 5
      ```

      ```bash
      # bash
      npc alicanto "What ethical considerations should guide AI development?" --max_facts_per_chain 0.5 --max_thematic_groups 3 --max_criticisms_per_group 3 max_conceptual_combinations 3 max_experiments 10 

      npc alicanto "What is the future of remote work?" --format report # NOTE: Report generation and formatting requires latex installed.
      ```
    - ## `/brainblast`: searching through past messages (soon to incorporate options for knowledge graph)
        ```bash
        # npcsh
        /brainblast 'subtle summer winds'  --top_k 10
        ```
        ```bash
        # bash
        npc brainblast 'executing a mirror in the wonderous moon'                                        
        ```
    - ## `/breathe`: Condense conversation context (shell only):
        ```bash
        # npcsh
        /breathe
        /breathe -p ollama -m qwen3:latest 
        ```
    - ## `/compile`: render npcs for use without re-loading npcsh
      ```bash
      # npcsh
      /compile ./npc_team/sibiji.npc      
      ```
    - ## `/flush`: flush context  (shell only):
      If you're in the NPC shell and have been in a conversation thats going nowhere and you want to start over... just flush theh contexf.
      ```bash
      /flush
      ```


    - ## `/guac`

    <p align="center"><a href ="https://github.com/npc-worldwide/npcsh/blob/main/docs/guac.md"> 
      <img src="https://raw.githubusercontent.com/npc-worldwide/npcsh/main/npcsh/npc_team/guac.png" alt="npcsh logo of a solarpunk sign", width=250></a>
    </p> 

    - a replacement shell for interpreters like python/r/node/julia with an avocado input marker 🥑 that brings a pomodoro-like approach to interactive coding.
    - available as a standalone program runnable via the `guac` command after `npcsh` has been installed via pip.
   
        - Simulation:      
            `🥑 Make a markov chain simulation of a random walk in 2D space with 1000 steps and visualize`
            ```
            # Generated python code:
            import numpy as np
            import matplotlib.pyplot as plt

            # Number of steps
            n_steps = 1000

            # Possible moves: up, down, left, right
            moves = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])

            # Initialize position array
            positions = np.zeros((n_steps+1, 2), dtype=int)

            # Generate random moves
            for i in range(1, n_steps+1):
                step = moves[np.random.choice(4)]
                positions[i] = positions[i-1] + step

            # Plot the random walk
            plt.figure(figsize=(8, 8))
            plt.plot(positions[:, 0], positions[:, 1], lw=1)
            plt.scatter([positions[0, 0]], [positions[0, 1]], color='green', label='Start')
            plt.scatter([positions[-1, 0]], [positions[-1, 1]], color='red', label='End')
            plt.title('2D Random Walk - 1000 Steps (Markov Chain)')
            plt.xlabel('X Position')
            plt.ylabel('Y Position')
            plt.legend()
            plt.grid(True)
            plt.axis('equal')
            plt.show()
            # Generated code executed successfully
          
            ```
            <p align="center">
              <img src="https://raw.githubusercontent.com/npc-worldwide/npcsh/main/test_data/markov_chain.png" alt="markov_chain_figure", width=250>
            </p>
            
            Access the variables created in the code:    
            `🥑 print(positions)`
            ```
            [[  0   0]
            [  0  -1]
            [ -1  -1]
            ...
            [ 29 -23]
            [ 28 -23]
            [ 27 -23]]
            ```
        
        - Run a python script:   
            `🥑 run file.py`    
        - Refresh:    
            `🥑 /refresh`       
        - Show current variables:    
            `🥑 /show`    

        A guac session progresses through a series of stages, each of equal length. Each stage adjusts the emoji input prompt. Once the stages have passed, it is time to refresh. Stage 1: `🥑`, Stage 2: `🥑🔪` Stage 3: `🥑🥣` Stage:4 `🥑🥣🧂`, `Stage 5: 🥘 TIME TO REFRESH`. At stage 5, the user is reminded to refresh with the /refresh macro. This will evaluate the session so farand suggest and implement new functions or automations that will aid in future sessions, with the ultimate approval of the user.


    - ## `/help`: Show help for commands, NPCs, or Jinxs. 
         ```bash
         /help 
         ```
         ```
         npc help
         ```
    - ## `/init` - Initialize NPC project    
        -set up bare bones infra for an npc team
        ```bash
        # npcsh 
        /init
        ```
        ```bash
        # bash 
        npc init
        ```
    

    - ## `/jinxs` : show available jinxs for team
        Jinxs are Jinja execution templates that let users develop small programs that can build on each other and reference each other through jinja templating. Jinx methods allow us to give smaller LLMs the scaffolding to perform `tool calling`, so to speak, reliably
        ```bash
        # npcsh 
        /jinxs
        # bash 
        npc jinxs
        ```

        ```python
        Available Jinxs:     
        --- Jinxs for NPC: sibiji ---                                                                                                                                                                                                           

        • /bash_executor: Execute bash queries.                                                                                                                                                                                                

        • /calc: A jinx to simplify and evaluate mathematical expressions   (/calc 1+5, /calc 47233*234234)                                                                                                                

        • /data_pull: Execute queries on the ~/npcsh_history.db to pull data. The database contains only information about conversations and other user-provided data. It does not store any information about individual files (/data_pull 'select * from conversation_history limit 10')


        • /file_editor: Examines a file, determines what changes are needed, and applies those changes. (/file_editor filename.py 'instructions for carrying out the editing')                                                                                                                                   

        • /image_generation_jinx: Generates images based on a text prompt. (/image_generation_jinx 'prompt for llm' output_name )                                                                                                                                       

        • /internet_search: Searches the web for information based on a query in order to verify timiely details (e.g. current events) or to corroborate information in uncertain situations. Should be mainly only used when users            
          specifically request a search, otherwise an LLMs basic knowledge should be sufficient. ( /internet_search 'cost of cubs tickets' )
        • /local_search: Searches files in current and downstream directories to find items related to the users query using fuzzy matching. (/local_search 'class NPC')
        Returns only relevant snippets (10 lines around matches) to avoid including too much irrelevant content. Intended for fuzzy searches, not for understanding file sizes.                                                                                                                                                                          

        • /python_executor: Execute scripts with python. Set the ultimate result as the "output" variable. It must be a string. Do not add unnecessary print statements. (/python_executor 'import numpy as np; print(np.arange(1000))')
        • /screen_capture_analysis_jinx: Captures the whole screen and sends the image for analysis  (mostly redundant with /ots.)  
        ```



    - ## `/ots`: Over-the-shoulder screen shot analysis
        - Screenshot analysis:     
        ```bash
        #npcsh
        /ots
        /ots output_filename =...
        ```
        ```bash
        #bash
        npc ots ...
        ```
    - ## `/plan`: set up cron jobs:
        ```bash
        # npcsh
        /plan 'set up a cron job that reminds me to stretch every thirty minutes' -m gemma3:27b -p ollama 
        ```
        ```bash
        # bash
        npc plan 'record my cpu usage percentage every 45 minutes' 
        ```

    - ## `/plonk`: Computer use:     
        ```bash
        # npcsh
        /plonk -n 'npc_name' -sp 'task for plonk to carry out '

        #bash
        npc plonk
        ```
    - ## `/pti`: a reasoning REPL loop with interruptions
      
        ```npcsh
        /pti  -n frederic -m qwen3:latest -p ollama 
        ```

        Or from the bash cmd line:
        ```bash
        pti
        ```
      <p align="center"><a href ="https://github.com/npc-worldwide/npcsh/blob/main/docs/pti.md"> 
        <img src="https://raw.githubusercontent.com/npc-worldwide/npcsh/main/npcsh/npc_team/frederic4.png" alt="npcsh logo of frederic the bear and the pti logo", width=250></a>
      </p>

    - ## `/rag`: embedding search through chroma db, optional file input
    - ## `/roll`: your video generation assistant
      - 
        ```npcsh
        /roll --provider ollama --model llama3
        ```

    - ## `/sample`: one-shot sampling from LLMs with specific parameters
        ```bash
        # npcsh
        /sample 'prompt'
        /sample -m gemini-1.5-flash "Summarize the plot of 'The Matrix' in three sentences."

        /sample --model claude-3-5-haiku-latest "Translate 'good morning' to Japanese."

        /sample model=qwen3:latest "tell me about the last time you went shopping."


        ```
        ```bash
        # bash
        npc sample -p ollama -m gemma3:12b --temp 1.8 --top_k 50 "Write a haiku about the command line."

        npc sample model=gpt-4o-mini "What are the primary colors?" --provider openai
        ```

    - ## `/search`: use an internet search provider     
        ```npcsh
        /search -sp perplexity 'cal bears football schedule'
        /search --sprovider duckduckgo 'beef tongue'        
        # Other search providers could be added, but we have only integrated duckduckgo and perplexity for the moment.
        ```

        ```bash
        npc search 'when is the moon gonna go away from the earth'
        ```
    

    - ## `/serve`: serve an npc team     
        ```bash
        /serve 
        /serve ....    
        # Other search providers could be added, but we have only integrated duckduckgo and perplexity for the moment.
        ```

        ```bash
        npc serve
        ```

    - ## `/set`: change current model, env params
        ```bash
        /set model ... 
        /set provider ...
        /set NPCSH_API_URL https://localhost:1937
        ```

        ```bash
        npc set ...
        ```
    - ## `/sleep`: prune and evolve the current knowledge graph 
        ```bash
        /sleep
        /sleep --dream
        /sleep --ops link_facts,deepen
        ```

        ```bash
        npc sleep
        ```
    - ## `/spool`
    <p align="center"><a href ="https://github.com/npc-worldwide/npcsh/blob/main/docs/spool.md"> 
      <img src="https://raw.githubusercontent.com/npc-worldwide/npcsh/main/npcsh/npc_team/spool.png" alt="logo for spool", width=250></a>
    </p>

    - Enter chat loop with isolated context, attachments, specified models/providers:     
        ```npcsh
        /spool -n <npc_name>
        /spool --attachments ./test_data/port5337.png,./test_data/yuan2004.pdf,./test_data/books.csv
        /spool --provider ollama --model llama3
        /spool -p deepseek -m deepseek-reasoner
        /spool -n alicanto
        ```



    - ## Trigger: schedule listeners, daemons
        ```bash
        /trigger 'a description of a trigger to implement with system daemons/file system listeners.' -m gemma3:27b -p ollama
        ```
        ```bash
        npc trigger
        ``` 




    

    - ## `/vixynt`: Image generation and editing:   
        ```bash
        npcsh
        /vixynt 'an image of a dog eating a hat'
        /vixynt --output_file ~/Desktop/dragon.png "A terrifying dragon"
        /vixynt "A photorealistic portrait of a cat wearing a wizard hat in the dungeon of the master and margarita" -w 1024.   height=1024        
        /vixynt -igp ollama  --igmodel Qwen/QwenImage --output_file /tmp/sub.png width=1024 height=512 "A detailed steampunk submarine exploring a vibrant coral reef, wide aspect ratio"
        ```

        ```bash
        # bash
        npc vixynt --attachments ./test_data/rabbit.PNG "Turn this rabbit into a fierce warrior in a snowy winter scene" -igp openai -igm gpt-image
        npc vixynt --igmodel CompVis/stable-diffusion-v1-4 --igprovider diffusers "sticker of a red tree"
        ```





    - ## `/wander`: daydreaming for LLMs

      <p align="center"><a href ="https://github.com/npc-worldwide/npcsh/blob/main/docs/wander.md">
        <img src="https://raw.githubusercontent.com/npc-worldwide/npcsh/main/npcsh/npc_team/kadiefa.png" alt="logo for wander", width=250></a>
      </p>
      A system for thinking outside of the box. From our testing, it appears gpt-4o-mini and gpt-series models in general appear to wander the most through various languages and ideas with high temperatures. Gemini models and many llama ones appear more stable despite high temps. Thinking models in general appear to be worse at this task.

      - Wander with an auto-generated environment  
        ```
        npc --model "gemini-2.0-flash"  --provider "gemini"  wander "how does the bar of a galaxy influence the the surrounding IGM?" \
          n-high-temp-streams=10 \
          high-temp=1.95 \
          low-temp=0.4 \
          sample-rate=0.5 \
          interruption-likelihood=1
        ```
      - Specify a custom environment
        ```
        npc --model "gpt-4o-mini"  --provider "openai"  wander "how does the goos-hanchen effect impact neutron scattering?" \
          environment='a ships library in the south.' \
          num-events=3 \
          n-high-temp-streams=10 \
          high-temp=1.95 \
          low-temp=0.4 \
          sample-rate=0.5 \
          interruption-likelihood=1
        ```
      - Control event generation
        ```
        npc wander "what is the goos hanchen effect and does it affect water refraction?" \
        --provider "ollama" \
        --model "deepseek-r1:32b" \
        environment="a vast, dark ocean ." \
        interruption-likelihood=.1
        ```

    - ## `/yap`: an agentic voice control loop


    <p align="center"><a href ="https://github.com/npc-worldwide/npcsh/blob/main/docs/yap.md"> 
      <img src="https://raw.githubusercontent.com/npc-worldwide/npcsh/main/npcsh/npc_team/yap.png" alt="logo for yap ", width=250></a>
    </p>

    - an agentic voice control loop with a specified agent. When launching `yap`, the user enters the typical `npcsh` agentic loop except that the system is waiting for either text or audio input.
    - voice chat:     
        ```bash
        # npcsh
        /yap
        ```
        ```bash
        # bash
        yap
        npc yap
        ```

## Inference Capabilities
- `npcsh` works with local and enterprise LLM providers through its LiteLLM integration, allowing users to run inference from Ollama, LMStudio, vLLM, MLX, OpenAI, Anthropic, Gemini, and Deepseek, making it a versatile tool for both simple commands and sophisticated AI-driven tasks. 

## Read the Docs

Read the docs at [npcsh.readthedocs.io](https://npcsh.readthedocs.io/en/latest/)


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
└── assembly_lines/    # Project workflows
    └── example.pipe
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
