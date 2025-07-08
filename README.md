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


## npcsh
- a bash-replacement shell (`npcsh`) that can process bash, natural language, or special macro calls. `npcsh` detects whether input is bash or natural language and processes it accordingly. 
    
    - Users can specify whether natural language commands are processed in one of three ways:
        - agentically (i.e. an NPC reviews and decides to pass to other NPCs or to use NPC tools called `jinxs` (short for Jinja Template Executions) to carry out tasks.
        - conversationally (the NPC generates a response which the user can approve to run) 
        - directly through bash execution (the NPC responds by generating executable bash code which is then processed automatically in the shell.
    
        Switching between the modes within the session is straightforward and the user can specify the default mode in the `.npcshrc` file described in greater detail below. The default mode is agentic, but the user can switch by typing `/chat` to switch to conversational mode or `/cmd` to switch to bash execution mode.
    
    - Web searching     
        ```
        /search -p perplexity 'cal bears football schedule'
        ```
    - One shot sampling 
        ```
        /sample 'prompt'
        ```

    - Image generation:      
        ```
        /vixynt 'an image of a dog eating a hat'
        ```
        
    - Process Identification:       
        ```    
        please identify the process consuming the most memory on my computer
        ```    
    - Screenshot analysis:     
        ```
        /ots
        ```
    - voice chat:     
        ```
        /yap
        ```
    - Computer use:     
        ```
        /plonk -n 'npc_name' -sp 'task for plonk to carry out '
        ```
    - Enter chat loop with an NPC:     
        ```
        /spool -n <npc_name>
        ```

## `guac`

<p align="center"><a href ="https://github.com/npc-worldwide/npcsh/blob/main/docs/guac.md"> 
  <img src="https://raw.githubusercontent.com/npc-worldwide/npcsh/main/npcsh/npc_team/guac.png" alt="npcsh logo of a solarpunk sign", width=250></a>
</p> 

- a replacement shell for interpreters like python/r/node/julia with an avocado input marker 🥑 that brings a pomodoro-like approach to interactive coding. 
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

 
## `npc`
- A command line interface offering the capabilities of the npc shell from a regular bash shell. Our mascot agent Sibiji the spider will help you weave your agent web with the `npc` CLI. 

<p align="center">
<img src="https://raw.githubusercontent.com/npc-worldwide/npcsh/main/npcsh/npc_team/sibiji.png" alt="npcsh logo with sibiji the spider">
</p>

- The NPC CLI lets users iterate and experiment with AI through bash commands. Below is a cheat sheet that shows how to use the `npc` CLI.

  - **Ask a Generic Question**
    ```bash
    npc 'has there ever been a better pasta shape than bucatini?'
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
    

  - **Compile an NPC**
    ```bash
    npc compile /path/to/npc.npc
    ```

  - **Computer Use**
    ```bash
    npc plonk -n 'npc_name' -sp 'task for plonk to carry out'
    ```

  - **Generate Image**
    ```bash
    npc vixynt 'generate an image of a rabbit eating ham in the brink of dawn' model='gpt-image-1' provider='openai'
    ```
  

  - **Search the Web**
    ```bash
    npc search -q "cal golden bears football schedule" -sp perplexity
    ```

  - **Serve an NPC Team**
    ```bash
    npc serve --port 5337 --cors='http://localhost:5137/'
    ```

  - **Screenshot Analysis**
    ```bash
    npc ots
    ```



## `alicanto` : a research exploration agent flow. 

<p align="center"><a href ="https://github.com/npc-worldwide/npcsh/blob/main/docs/alicanto.md"> 
  <img src="https://raw.githubusercontent.com/npc-worldwide/npcsh/main/npcsh/npc_team/alicanto.png" alt="logo for deep research", width=250></a>
</p>

  - Examples:
    ```
    npc alicanto "What are the implications of quantum computing for cybersecurity?"
    ```

    - With more researchers and deeper exploration
    
    ```
    npc alicanto "How might climate change impact global food security?" --num-npcs 8 --depth 5

    ```
    - Control exploration vs. exploitation balance

    ```
    npc alicanto "What ethical considerations should guide AI development?" --exploration 0.5

    ```
    - Different output formats
    ```    
    npc alicanto "What is the future of remote work?" --format report
    ```
 
## `pti`
-  a reasoning REPL loop with explicit checks to request inputs from users following thinking traces.
 
<p align="center"><a href ="https://github.com/npc-worldwide/npcsh/blob/main/docs/pti.md"> 
  <img src="https://raw.githubusercontent.com/npc-worldwide/npcsh/main/npcsh/npc_team/frederic4.png" alt="npcsh logo of frederic the bear and the pti logo", width=250></a>
</p>
Speak with frederic the bear who, once he's done thinking, asks you for input before trudging on so it can work with confidence.

```bash
pti
```


## `spool`
- a simple agentic REPL chat loop with a specified agent.

<p align="center"><a href ="https://github.com/npc-worldwide/npcsh/blob/main/docs/spool.md"> 
  <img src="https://raw.githubusercontent.com/npc-worldwide/npcsh/main/npcsh/npc_team/spool.png" alt="logo for spool", width=250></a>
</p>

## `yap`


<p align="center"><a href ="https://github.com/npc-worldwide/npcsh/blob/main/docs/yap.md"> 
  <img src="https://raw.githubusercontent.com/npc-worldwide/npcsh/main/npcsh/npc_team/yap.png" alt="logo for yap ", width=250></a>
</p>

- an agentic voice control loop with a specified agent. When launching `yap`, the user enters the typical `npcsh` agentic loop except that the system is waiting for either text or audio input.

```
yap 
```


## `wander` 

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





## Inference Capabilities
- `npcsh` works with local and enterprise LLM providers through its LiteLLM integration, allowing users to run inference from Ollama, LMStudio, OpenAI, Anthropic, Gemini, and Deepseek, making it a versatile tool for both simple commands and sophisticated AI-driven tasks. 

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
