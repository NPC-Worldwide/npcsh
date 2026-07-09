# npcsh Benchmarks

The benchmark suite measures how well a model can drive `npcsh` as an agentic shell. It covers 135 tasks across 15 categories, from basic shell commands and file operations to multi-step workflows, debugging, tool chaining, delegation, web search, and media generation. Each task is scored pass/fail by an automated verifier.

Run the benchmark yourself:

```bash
# Local model via Ollama
python -m npcsh.benchmark.rust_runner --model qwen3.5:9b --provider ollama

# API model
python -m npcsh.benchmark.rust_runner --model gemini-2.5-flash --provider gemini

# Single category or task
python -m npcsh.benchmark.rust_runner --category shell --limit 10
python -m npcsh.benchmark.rust_runner --task-id shell-01
```

Results are written to `~/.npcsh/benchmarks/local/`.

## Overall Scores

| Family | Model | Score |
|--------|-------|-------|
| **Kimi** | k2.5 | **121/125 (97%)** |
| **Qwen3.5** | 0.8b | 31/125 (24%) |
| | 2b | 81/125 (65%) |
| | 4b | 77/125 (62%) |
| | 9b | **100/125 (80%)** |
| | 35b | **111/125 (88%)** |
| | 397b | **120/125 (96%)** |
| **Qwen3** | 0.6b | — |
| | 1.7b | 42/125 (34%) |
| | 4b | **94/125 (75%)** |
| | 8b | 85/125 (68%) |
| | 30b | **103/125 (82%)** |
| **Gemma4** | e4b | 34/125 (27%) |
| | 31b | **105/125 (84%)** |
| **Gemma3** | 1b | — |
| | 4b | 37/125 (30%) |
| | 12b | 77/125 (62%) |
| | 27b | 73/125 (58%) |
| **Llama** | 3.2:1b | — |
| | 3.2:3b | 26/125 (20%) |
| | 3.1:8b | 60/125 (48%) |
| **Mistral** | small3.2 | 72/125 (57%) |
| | ministral-3 | 51/125 (40%) |
| | large-3 | 59/125 (47%) |
| **Devstral** | 2 | 60/125 (48%) |
| **MiniMax** | M2.7 | **120/125 (96%)** |
| **Phi** | phi4 | 58/125 (46%) |
| **GPT-OSS** | 20b | 94/125 (75%) |
| **OLMo2** | 7b | 13/125 (10%) |
| | 13b | 47/125 (38%) |
| **Cogito** | 3b | 10/125 (8%) |
| **GLM** | 4.7-flash | **102/125 (82%)** |
| | 5 | **120/125 (96%)** |
| **Nemotron** | 3-super | 49/125 (39%) |
| **Gemini** | 2.5-flash | — |
| | 3.1-flash | — |
| | 3.1-pro | — |
| **Claude** | 4.6-sonnet | — |
| | 4.5-haiku | — |
| **GPT** | 5-mini | — |
| **DeepSeek** | v4-flash | **99/125 (79%)** |
| | chat | — |
| | reasoner | — |

## Category Breakdown

<table>
<tr>
<th rowspan="2">Category</th>
<th colspan="4">Qwen3.5</th>
<th colspan="5">Qwen3</th>
<th colspan="2">Gemma4</th>
<th colspan="3">Gemma3</th>
<th>Llama</th>
<th colspan="3">Mistral</th>
<th>Phi</th>
<th>GPT-OSS</th>
<th>Cogito</th>
<th colspan="2">GLM</th>
<th>Kimi</th>
<th>Qwen3.5</th>
<th>MiniMax</th>
<th>Devstral</th>
<th>Nemotron</th>
<th>DeepSeek</th>
</tr>
<tr>
<th>0.8b</th><th>2b</th><th>9b</th><th>35b</th>
<th>1.7b</th><th>4b</th><th>8b</th><th>30b</th><th>0.6b</th>
<th>e4b</th><th>31b</th>
<th>4b</th><th>12b</th><th>27b</th>
<th>3.2:3b</th>
<th>small3.2</th><th>ministral-3</th><th>large-3</th>
<th>phi4</th>
<th>20b</th>
<th>3b</th>
<th>4.7</th><th>5</th>
<th>k2.5</th>
<th>397b</th>
<th>M2.7</th>
<th>2</th>
<th>3-super</th>
<th>v4-flash</th>
</tr>
<tr><td>shell (10)</td><td>5</td><td>6</td><td>10</td><td>10</td><td>8</td><td>8</td><td>9</td><td>9</td><td>—</td><td>6</td><td>10</td><td>6</td><td>6</td><td>9</td><td>6</td><td>10</td><td>7</td><td>8</td><td>9</td><td>10</td><td>0</td><td>10</td><td>10</td><td>10</td><td>10</td><td>10</td><td>9</td><td>7</td><td>8</td></tr>
<tr><td>file-ops (10)</td><td>8</td><td>9</td><td>10</td><td>10</td><td>8</td><td>10</td><td>9</td><td>10</td><td>—</td><td>5</td><td>8</td><td>6</td><td>9</td><td>10</td><td>2</td><td>6</td><td>10</td><td>8</td><td>10</td><td>10</td><td>0</td><td>10</td><td>9</td><td>10</td><td>10</td><td>9</td><td>10</td><td>5</td><td>9</td></tr>
<tr><td>python (10)</td><td>0</td><td>3</td><td>9</td><td>10</td><td>0</td><td>5</td><td>6</td><td>6</td><td>—</td><td>1</td><td>10</td><td>0</td><td>3</td><td>1</td><td>0</td><td>3</td><td>6</td><td>6</td><td>4</td><td>10</td><td>0</td><td>10</td><td>10</td><td>10</td><td>10</td><td>10</td><td>6</td><td>5</td><td>10</td></tr>
<tr><td>data (10)</td><td>0</td><td>2</td><td>4</td><td>6</td><td>2</td><td>4</td><td>5</td><td>6</td><td>—</td><td>0</td><td>9</td><td>1</td><td>5</td><td>7</td><td>0</td><td>5</td><td>9</td><td>5</td><td>4</td><td>6</td><td>0</td><td>5</td><td>9</td><td>9</td><td>9</td><td>9</td><td>4</td><td>4</td><td>9</td></tr>
<tr><td>system (10)</td><td>2</td><td>8</td><td>9</td><td>10</td><td>7</td><td>9</td><td>7</td><td>10</td><td>—</td><td>6</td><td>10</td><td>5</td><td>9</td><td>7</td><td>2</td><td>9</td><td>6</td><td>10</td><td>6</td><td>9</td><td>0</td><td>10</td><td>10</td><td>10</td><td>10</td><td>10</td><td>8</td><td>6</td><td>9</td></tr>
<tr><td>text (10)</td><td>1</td><td>7</td><td>6</td><td>8</td><td>2</td><td>10</td><td>6</td><td>7</td><td>—</td><td>0</td><td>8</td><td>3</td><td>9</td><td>8</td><td>1</td><td>7</td><td>0</td><td>0</td><td>4</td><td>8</td><td>0</td><td>7</td><td>10</td><td>10</td><td>10</td><td>10</td><td>0</td><td>0</td><td>10</td></tr>
<tr><td>debug (10)</td><td>2</td><td>6</td><td>10</td><td>10</td><td>0</td><td>4</td><td>2</td><td>10</td><td>—</td><td>4</td><td>0</td><td>0</td><td>3</td><td>0</td><td>0</td><td>4</td><td>0</td><td>2</td><td>0</td><td>9</td><td>0</td><td>9</td><td>10</td><td>10</td><td>10</td><td>10</td><td>0</td><td>3</td><td>10</td></tr>
<tr><td>git (10)</td><td>0</td><td>8</td><td>6</td><td>9</td><td>2</td><td>9</td><td>9</td><td>8</td><td>—</td><td>2</td><td>9</td><td>4</td><td>6</td><td>9</td><td>4</td><td>8</td><td>4</td><td>0</td><td>6</td><td>8</td><td>0</td><td>5</td><td>10</td><td>10</td><td>9</td><td>9</td><td>0</td><td>1</td><td>9</td></tr>
<tr><td>multi-step (10)</td><td>0</td><td>6</td><td>7</td><td>6</td><td>0</td><td>6</td><td>3</td><td>7</td><td>—</td><td>0</td><td>9</td><td>3</td><td>5</td><td>5</td><td>2</td><td>3</td><td>0</td><td>0</td><td>5</td><td>4</td><td>0</td><td>5</td><td>8</td><td>9</td><td>9</td><td>9</td><td>2</td><td>0</td><td>4</td></tr>
<tr><td>scripting (10)</td><td>1</td><td>5</td><td>8</td><td>10</td><td>0</td><td>7</td><td>2</td><td>6</td><td>—</td><td>0</td><td>9</td><td>0</td><td>2</td><td>1</td><td>0</td><td>3</td><td>1</td><td>6</td><td>3</td><td>7</td><td>0</td><td>8</td><td>10</td><td>10</td><td>10</td><td>10</td><td>8</td><td>2</td><td>5</td></tr>
<tr><td>image-gen (5)</td><td>5</td><td>5</td><td>5</td><td>5</td><td>5</td><td>5</td><td>5</td><td>5</td><td>—</td><td>5</td><td>5</td><td>3</td><td>5</td><td>3</td><td>5</td><td>5</td><td>1</td><td>5</td><td>2</td><td>5</td><td>5</td><td>5</td><td>5</td><td>5</td><td>5</td><td>5</td><td>5</td><td>5</td><td>0</td></tr>
<tr><td>audio-gen (5)</td><td>5</td><td>4</td><td>5</td><td>5</td><td>5</td><td>5</td><td>5</td><td>5</td><td>—</td><td>5</td><td>5</td><td>4</td><td>5</td><td>5</td><td>4</td><td>5</td><td>1</td><td>5</td><td>5</td><td>5</td><td>5</td><td>5</td><td>5</td><td>5</td><td>5</td><td>5</td><td>5</td><td>5</td><td>5</td></tr>
<tr><td>web-search (5)</td><td>1</td><td>5</td><td>4</td><td>5</td><td>1</td><td>5</td><td>4</td><td>5</td><td>—</td><td>0</td><td>5</td><td>1</td><td>5</td><td>5</td><td>0</td><td>4</td><td>5</td><td>0</td><td>0</td><td>3</td><td>0</td><td>5</td><td>5</td><td>5</td><td>5</td><td>5</td><td>0</td><td>2</td><td>5</td></tr>
<tr><td>delegation (5)</td><td>0</td><td>2</td><td>3</td><td>3</td><td>0</td><td>2</td><td>2</td><td>4</td><td>—</td><td>0</td><td>4</td><td>0</td><td>2</td><td>0</td><td>0</td><td>0</td><td>0</td><td>3</td><td>0</td><td>0</td><td>3</td><td>4</td><td>4</td><td>4</td><td>4</td><td>2</td><td>3</td><td>1</td></tr>
<tr><td>tool-chain (5)</td><td>1</td><td>5</td><td>4</td><td>4</td><td>2</td><td>5</td><td>2</td><td>5</td><td>—</td><td>0</td><td>4</td><td>1</td><td>3</td><td>3</td><td>0</td><td>0</td><td>1</td><td>1</td><td>0</td><td>0</td><td>5</td><td>5</td><td>5</td><td>5</td><td>5</td><td>1</td><td>1</td><td>5</td></tr>
<tr><td><b>Total (125)</b></td><td><b>31</b></td><td><b>81</b></td><td><b>100</b></td><td><b>111</b></td><td><b>42</b></td><td><b>94</b></td><td><b>76</b></td><td><b>103</b></td><td>—</td><td><b>34</b></td><td><b>105</b></td><td><b>37</b></td><td><b>77</b></td><td><b>73</b></td><td><b>26</b></td><td><b>72</b></td><td><b>51</b></td><td><b>59</b></td><td><b>58</b></td><td><b>94</b></td><td><b>10</b></td><td><b>102</b></td><td><b>120</b></td><td><b>121</b></td><td><b>120</b></td><td><b>120</b></td><td><b>60</b></td><td><b>49</b></td><td><b>99</b></td></tr>
</table>
