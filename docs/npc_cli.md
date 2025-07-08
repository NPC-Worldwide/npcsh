
## NPC CLI
When npcsh is installed, it comes with the `npc` cli as well. The `npc` cli has various command to make initializing and serving NPC projects easier.

Users can make queries like so:
```bash
$ npc 'whats the biggest filei  n my computer'
Loaded .env file from /home/caug/npcww/npcsh
action chosen: request_input
explanation given: The user needs to provide more context about their operating system or specify which directory to search for the biggest file.

Additional input needed: The user did not specify their operating system or the directory to search for the biggest file, making it unclear how to execute the command.
Please specify your operating system (e.g., Windows, macOS, Linux) and the directory you want to search in.: linux and root
action chosen: execute_command
explanation given: The user is asking for the biggest file on their computer, which can be accomplished with a simple bash command that searches for the largest files.
sibiji generating command
LLM suggests the following bash command: sudo find / -type f -exec du -h {} + | sort -rh | head -n 1
Running command: sudo find / -type f -exec du -h {} + | sort -rh | head -n 1
Command executed with output: 11G       /home/caug/.cache/huggingface/hub/models--state-spaces--mamba-2.8b/blobs/39911a8470a2b256016b57cc71c68e0f96751cba5b229216ab1f4f9d82096a46

I ran a command on your Linux system that searches for the largest files on your computer. The command `sudo find / -type f -exec du -h {} + | sort -rh | head -n 1` performs the following steps:

1. **Find Command**: It searches for all files (`-type f`) starting from the root directory (`/`).
2. **Disk Usage**: For each file found, it calculates its disk usage in a human-readable format (`du -h`).
3. **Sort**: It sorts the results in reverse order based on size (`sort -rh`), so the largest files appear first.
4. **Head**: Finally, it retrieves just the largest file using `head -n 1`.

The output indicates that the biggest file on your system is located at `/home/caug/.cache/huggingface/hub/models--state-spaces--mamba-2.8b/blobs/39911a8470a2b256016b57cc71c68e0f96751cba5b229216ab1f4f9d82096a46` and is 11GB in size.

```

```bash
$ npc 'whats the weather in tokyo'
Loaded .env file from /home/caug/npcww/npcsh
action chosen: invoke_jinx
explanation given: The user's request for the current weather in Tokyo requires up-to-date information, which can be best obtained through an internet search.
Jinx found: internet_search
Executing jinx with input values: {'query': 'whats the weather in tokyo'}
QUERY in jinx whats the weather in tokyo
[{'title': 'Tokyo, Tokyo, Japan Weather Forecast | AccuWeather', 'href': 'https://www.accuweather.com/en/jp/tokyo/226396/weather-forecast/226396', 'body': 'Tokyo, Tokyo, Japan Weather Forecast, with current conditions, wind, air quality, and what to expect for the next 3 days.'}, {'title': 'Tokyo, Japan 14 day weather forecast - timeanddate.com', 'href': 'https://www.timeanddate.com/weather/japan/tokyo/ext', 'body': 'Tokyo Extended Forecast with high and low temperatures. °F. Last 2 weeks of weather'}, {'title': 'Tokyo, Tokyo, Japan Current Weather | AccuWeather', 'href': 'https://www.accuweather.com/en/jp/tokyo/226396/current-weather/226396', 'body': 'Current weather in Tokyo, Tokyo, Japan. Check current conditions in Tokyo, Tokyo, Japan with radar, hourly, and more.'}, {'title': 'Weather in Tokyo, Japan - timeanddate.com', 'href': 'https://www.timeanddate.com/weather/japan/tokyo', 'body': 'Current weather in Tokyo and forecast for today, tomorrow, and next 14 days'}, {'title': 'Tokyo Weather Forecast Today', 'href': 'https://japanweather.org/tokyo', 'body': "For today's mild weather in Tokyo, with temperatures between 13ºC to 16ºC (55.4ºF to 60.8ºF), consider wearing: - Comfortable jeans or slacks - Sun hat (if spending time outdoors) - Lightweight sweater or cardigan - Long-sleeve shirt or blouse. Temperature. Day. 14°C. Night. 10°C. Morning. 10°C. Afternoon."}] <class 'list'>
RESULTS in jinx ["Tokyo, Tokyo, Japan Weather Forecast, with current conditions, wind, air quality, and what to expect for the next 3 days.\n Citation: https://www.accuweather.com/en/jp/tokyo/226396/weather-forecast/226396\n\n\n\nTokyo Extended Forecast with high and low temperatures. °F. Last 2 weeks of weather\n Citation: https://www.timeanddate.com/weather/japan/tokyo/ext\n\n\n\nCurrent weather in Tokyo, Tokyo, Japan. Check current conditions in Tokyo, Tokyo, Japan with radar, hourly, and more.\n Citation: https://www.accuweather.com/en/jp/tokyo/226396/current-weather/226396\n\n\n\nCurrent weather in Tokyo and forecast for today, tomorrow, and next 14 days\n Citation: https://www.timeanddate.com/weather/japan/tokyo\n\n\n\nFor today's mild weather in Tokyo, with temperatures between 13ºC to 16ºC (55.4ºF to 60.8ºF), consider wearing: - Comfortable jeans or slacks - Sun hat (if spending time outdoors) - Lightweight sweater or cardigan - Long-sleeve shirt or blouse. Temperature. Day. 14°C. Night. 10°C. Morning. 10°C. Afternoon.\n Citation: https://japanweather.org/tokyo\n\n\n", 'https://www.accuweather.com/en/jp/tokyo/226396/weather-forecast/226396\n\nhttps://www.timeanddate.com/weather/japan/tokyo/ext\n\nhttps://www.accuweather.com/en/jp/tokyo/226396/current-weather/226396\n\nhttps://www.timeanddate.com/weather/japan/tokyo\n\nhttps://japanweather.org/tokyo\n']
The current weather in Tokyo, Japan is mild, with temperatures ranging from 13°C to 16°C (approximately 55.4°F to 60.8°F). For today's conditions, it is suggested to wear comfortable jeans or slacks, a lightweight sweater or cardigan, and a long-sleeve shirt or blouse, especially if spending time outdoors. The temperature today is expected to reach a high of 14°C (57.2°F) during the day and a low of 10°C (50°F) at night.

For more detailed weather information, you can check out the following sources:
- [AccuWeather Forecast](https://www.accuweather.com/en/jp/tokyo/226396/weather-forecast/226396)
- [Time and Date Extended Forecast](https://www.timeanddate.com/weather/japan/tokyo/ext)
- [Current Weather on AccuWeather](https://www.accuweather.com/en/jp/tokyo/226396/current-weather/226396)
- [More on Time and Date](https://www.timeanddate.com/weather/japan/tokyo)
- [Japan Weather](https://japanweather.org/tokyo)
```


## Serving
To serve an NPC project, first install redis-server and start it

on Ubuntu:
```bash
sudo apt update && sudo apt install redis-server
redis-server
```

on macOS:
```bash
brew install redis
redis-server
```
Then navigate to the project directory and run:

```bash
npc serve
```
If you want to specify a certain port, you can do so with the `-p` flag:
```bash
npc serve -p 5337
```
or with the `--port` flag:
```bash
npc serve --port 5337

```
If you want to initialize a project based on templates and then make it available for serving, you can do so like this
```bash
npc serve -t 'sales, marketing' -ctx 'im developing a team that will focus on sales and marketing within the logging industry. I need a team that can help me with the following: - generate leads - create marketing campaigns - build a sales funnel - close deals - manage customer relationships - manage sales pipeline - manage marketing campaigns - manage marketing budget' -m llama3.2 -pr ollama
```
This will use the specified model and provider to generate a team of npcs to fit the templates and context provided.


Once the server is up and running, you can access the API endpoints at `http://localhost:5337/api/`. Here are some example curl commands to test the endpoints:

```bash
echo "Testing health endpoint..."
curl -s http://localhost:5337/api/health | jq '.'

echo -e "\nTesting execute endpoint..."
curl -s -X POST http://localhost:5337/api/execute \
  -H "Content-Type: application/json" \
  -d '{"commandstr": "hello world", "currentPath": "~/", "conversationId": "test124"}' | jq '.'

echo -e "\nTesting conversations endpoint..."
curl -s "http://localhost:5337/api/conversations?path=/tmp" | jq '.'

echo -e "\nTesting conversation messages endpoint..."
curl -s http://localhost:5337/api/conversation/test123/messages | jq '.'
```

## Planned cli features


* **Planned:** -npc scripts
-npc run select +sql_model   <run up>
-npc run select +sql_model+  <run up and down>
-npc run select sql_model+  <run down>
-npc run line <assembly_line>
-npc conjure fabrication_plan.fab



# Macros

While npcsh can decide the best option to use based on the user's input, the user can also execute certain actions with a macro. Macros are commands within the NPC shell that start with a forward slash (/) and are followed (in some cases) by the relevant arguments for those macros. Each macro is also available as a sub-program within the NPC CLI. In the following examples we demonstrate how to carry out the same operations from within npcsh and from a regular shell.


To learn about the available macros from within the shell, type:
```npcsh
npcsh> /help
```

or from bash
```bash
npc --help
#alternatively
npc -h
```