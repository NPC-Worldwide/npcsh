from npcpy.data.image import capture_screenshot
import time
import platform
from npcpy.llm_funcs import get_llm_response
from npcpy.work.desktop import perform_action, action_space
from PIL import Image, ImageDraw, ImageFont

def get_system_examples():
    system = platform.system()
    if system == "Windows":
        return "Examples: start firefox, notepad, calc, explorer"
    elif system == "Darwin":
        return "Examples: open -a Firefox, open -a TextEdit, open -a Calculator"
    else:
        return "Examples: firefox &, gedit &, gnome-calculator &"

def execute_plonk_command(request, action_space, model, provider, npc=None, max_iterations=10, debug=False):
    synthesized_summary = []

    """Synthesizes information gathered during the computer use run and logs key data points for
    analysis. This function can be extended to store or report the synthesized knowledge as required.
    """

    system = platform.system()
    system_examples = get_system_examples()
    
    messages = []
    last_action_feedback = "None"
    last_click_coords = None

    iteration_count = 0
    while iteration_count < max_iterations:
        # Gathering summary of actions performed this iteration
        synthesized_info = {
            'iteration': iteration_count + 1,
            'last_action_feedback': last_action_feedback,
            'last_click_coords': last_click_coords
        }
        synthesized_summary.append(synthesized_info)

        if debug:
            print(f"Synthesized info at iteration {iteration_count + 1}: {synthesized_info}")

        if debug:
            print(f"Iteration {iteration_count + 1}/{max_iterations}")

        # YOUR PROMPT, UNTOUCHED
        prompt_template = f"""
        Goal: {request}
        Feedback from last action: {last_action_feedback}

        Your task is to control the computer to achieve the goal.
        
        THOUGHT PROCESS:
        1. Analyze the screen. Is the application I need (e.g., a web browser) already open?
        2. If YES, `click` it. If NO, use `bash` to launch it. Use the examples: {system_examples}.
        
        
        CRITICAL COMPLETION RULE:
        Once the goal is visually complete on the screen, your ONLY next action is to use the 'quit' action.

        Your response MUST be a JSON object with an "actions" key.
        All clicking actions should use percentage coordinates relative 
        to the screen size, as we will
        manually translate them to the proper screen size. 
        your x and y values for clicks must ALWAYS be between 0 and 100.
        The x and y are (0,0) at the TOP LEFT CORNER OF THE SCREEN.
        The bottom right corner of the screen is (100,100).
        the bottom left corner is (0,100) and the top right corner is (100,0).
        
        
        
        
        ---
        EXAMPLE 1: Task "Create and save a file named 'memo.txt' with the text 'Meeting at 3pm'"
        {{
          "actions": [
            {{ "type": "bash", "command": "gedit &" }},
            {{ "type": "wait", "duration": 2 }},
            {{'type':'click', 'x': 10, 'y': 30}},  
            {{ "type": "type", "text": "Meeting at 3pm" }},
            {{ "type": "hotkey", "keys": ["ctrl", "s"] }},
            {{ "type": "wait", "duration": 1 }},
            {{ "type": "type", "text": "memo.txt" }},
            {{ "type": "key", "keys": ["enter"] }},
          ]
        }}
        ---
        EXAMPLE 2: Task "Search for news about space exploration"
        {{
          "actions": [
            {{ "type": "bash", "command": "firefox &" }},
            {{ "type": "wait", "duration": 3 }},
            {{ "type": "type", "text": "news about space exploration" }},
            {{ "type": "key", "keys": ["enter"] }},
          ]
        }}
        
        ---
        
        Once a task has been verified and completed, your action list should only be 
        {{
          "actions": [
            {{ "type": "quit" }}
          ]
        }}
        """

        screenshot_path = capture_screenshot(npc=npc, full=True).get('file_path')
        if not screenshot_path:
            time.sleep(2)
            continue

        image_to_send_path = screenshot_path
        if last_click_coords:
            try:
                img = Image.open(screenshot_path)
                draw = ImageDraw.Draw(img)
                width, height = img.size
                x_pixel = int(last_click_coords['x'] * width / 100)
                y_pixel = int(last_click_coords['y'] * height / 100)
                
                try:
                    font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=48)
                except IOError:
                    font = ImageFont.load_default()

                draw.text((x_pixel - 8, y_pixel - 12),
                          f"+{last_click_coords['x'],last_click_coords['y']}",
                          fill="red",
                          font=font)
                
                marked_image_path = "/tmp/marked_screenshot.png"
                img.save(marked_image_path)
                image_to_send_path = marked_image_path
                print(f"Drew marker at ({x_pixel}, {y_pixel}) on new screenshot.")
            except Exception as e:
                print(f"Failed to draw marker on image: {e}")
        
        response = get_llm_response(
            prompt=prompt_template,
            model=model,
            provider=provider,
            npc=npc,
            images=[image_to_send_path],
            messages=messages,
            format="json",
        )

        if "messages" in response:
            messages = response["messages"]
        
        response_data = response.get('response')

        if not isinstance(response_data, dict) or "actions" not in response_data:
            last_action_feedback = f"Invalid JSON response from model: {response_data}"
            continue

        actions_list = response_data.get("actions", [])
        
        if not isinstance(actions_list, list):
            last_action_feedback = "Model did not return a list in the 'actions' key."
            continue
        
        # Reset last click before processing new actions
        last_click_coords = None
        for action in actions_list:
            if debug:
                print(f"Executing action: {action}")
            if action.get("type") == "quit":
                print("Task complete: Model returned 'quit' action.")
                return "SUCCESS"
                
            result = perform_action(action)
            last_action_feedback = result.get("message") or result.get("output")

            if action.get("type") == "click":
                last_click_coords = {"x": action.get("x"), "y": action.get("y")}
            
            if result.get("status") == "error":
                print(f"Action failed, providing feedback to model: {last_action_feedback}")
                break 
            time.sleep(1)
        
        if not actions_list:
            last_action_feedback = "No actions were returned. The task is likely not complete. Re-evaluating."
            print(last_action_feedback)
        
        iteration_count += 1
    
    return None

def synthesize_and_display_summary(synthesized_summary, debug=False):
    """Synthesizes information gathered during the computer use run and logs key data points."""
    if not synthesized_summary:
        print("No synthesized info to display.")
        return

    print("\nSynthesized Summary of Computer Use Run:")
    for info in synthesized_summary:
        print(f"Iteration {info['iteration']}:\n"
              f"  Last Action Feedback: {info['last_action_feedback']}\n"
              f"  Last Click Coordinates: {info['last_click_coords']}")
    print("End of synthesized summary.\n")



def repl_loop():
    print("Assistant REPL - Type your plonk command or 'exit' to quit.")
    while True:
        user_input = input("Enter your command: ").strip()
        if user_input.lower() == 'exit':
            print("Exiting REPL. Goodbye!")
            break
        if not user_input:
            continue

        # Run the plonk command and get synthesized summary
        synthesized_summary = execute_plonk_command(
            request=user_input,
            action_space=action_space,
            model="gpt-4o-mini",
            provider="openai",
            max_iterations=8,
            debug=True
        )

        if synthesized_summary and isinstance(synthesized_summary, list):
            print("Command executed with synthesized summary.")
            synthesize_and_display_summary(synthesized_summary)
        else:
            print("Command did not complete within iteration limit or returned no summary.")


def execute_plonk_command(request, action_space, model, provider, npc=None, max_iterations=10, debug=False):
    """Synthesizes information gathered during the computer use run and logs key data points for
    analysis. This function can be extended to store or report the synthesized knowledge as required.
    """

    system = platform.system()
    system_examples = get_system_examples()
    
    messages = []
    last_action_feedback = "None"
    last_click_coords = None

    iteration_count = 0

    synthesized_summary = []

    while iteration_count < max_iterations:
        synthesized_info = {
            'iteration': iteration_count + 1,
            'last_action_feedback': last_action_feedback,
            'last_click_coords': last_click_coords
        }
        synthesized_summary.append(synthesized_info)

        if debug:
            print(f"Synthesized info at iteration {iteration_count + 1}: {synthesized_info}")

        if debug:
            print(f"Iteration {iteration_count + 1}/{max_iterations}")

        prompt_template = f"""
        Goal: {request}
        Feedback from last action: {last_action_feedback}

        Your task is to control the computer to achieve the goal.
        
        THOUGHT PROCESS:
        1. Analyze the screen. Is the application I need (e.g., a web browser) already open?
        2. If YES, `click` it. If NO, use `bash` to launch it. Use the examples: {system_examples}.
        
        
        CRITICAL COMPLETION RULE:
        Once the goal is visually complete on the screen, your ONLY next action is to use the 'quit' action.

        Your response MUST be a JSON object with an "actions" key.
        All clicking actions should use percentage coordinates relative 
        to the screen size, as we will
        manually translate them to the proper screen size. 
        your x and y values for clicks must ALWAYS be between 0 and 100.
        The x and y are (0,0) at the TOP LEFT CORNER OF THE SCREEN.
        The bottom right corner of the screen is (100,100).
        the bottom left corner is (0,100) and the top right corner is (100,0).
        
        
        
        ---
        EXAMPLE 1: Task "Create and save a file named 'memo.txt' with the text 'Meeting at 3pm'"
        {{
          "actions": [
            {{ "type": "bash", "command": "gedit &" }},
            {{ "type": "wait", "duration": 2 }},
            {{'type':'click', 'x': 10, 'y': 30}},  
            {{ "type": "type", "text": "Meeting at 3pm" }},
            {{ "type": "hotkey", "keys": ["ctrl", "s"] }},
            {{ "type": "wait", "duration": 1 }},
            {{ "type": "type", "text": "memo.txt" }},
            {{ "type": "key", "keys": ["enter"] }},
          ]
        }}
        ---
        EXAMPLE 2: Task "Search for news about space exploration"
        {{
          "actions": [
            {{ "type": "bash", "command": "firefox &" }},
            {{ "type": "wait", "duration": 3 }},
            {{ "type": "type", "text": "news about space exploration" }},
            {{ "type": "key", "keys": ["enter"] }},
          ]
        }}
        
        ---
        
        Once a task has been verified and completed, your action list should only be 
        {{
          "actions": [
            {{ "type": "quit" }}
          ]
        }}
        """

        screenshot_path = capture_screenshot(npc=npc, full=True).get('file_path')
        if not screenshot_path:
            time.sleep(2)
            continue

        image_to_send_path = screenshot_path
        if last_click_coords:
            try:
                img = Image.open(screenshot_path)
                draw = ImageDraw.Draw(img)
                width, height = img.size
                x_pixel = int(last_click_coords['x'] * width / 100)
                y_pixel = int(last_click_coords['y'] * height / 100)
                
                try:
                    font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=48)
                except IOError:
                    font = ImageFont.load_default()

                draw.text((x_pixel - 8, y_pixel - 12),
                          f"+{last_click_coords['x'],last_click_coords['y']}",
                          fill="red",
                          font=font)
                
                marked_image_path = "/tmp/marked_screenshot.png"
                img.save(marked_image_path)
                image_to_send_path = marked_image_path
                print(f"Drew marker at ({x_pixel}, {y_pixel}) on new screenshot.")
            except Exception as e:
                print(f"Failed to draw marker on image: {e}")
        
        response = get_llm_response(
            prompt=prompt_template,
            model=model,
            provider=provider,
            npc=npc,
            images=[image_to_send_path],
            messages=messages,
            format="json",
        )

        if "messages" in response:
            messages = response["messages"]
        
        response_data = response.get('response')

        if not isinstance(response_data, dict) or "actions" not in response_data:
            last_action_feedback = f"Invalid JSON response from model: {response_data}"
            continue

        actions_list = response_data.get("actions", [])
        
        if not isinstance(actions_list, list):
            last_action_feedback = "Model did not return a list in the 'actions' key."
            continue
        
        last_click_coords = None
        for action in actions_list:
            if debug:
                print(f"Executing action: {action}")
            if action.get("type") == "quit":
                print("Task complete: Model returned 'quit' action.")
                return synthesized_summary
                
            result = perform_action(action)
            last_action_feedback = result.get("message") or result.get("output")

            if action.get("type") == "click":
                last_click_coords = {"x": action.get("x"), "y": action.get("y")}
            
            if result.get("status") == "error":
                print(f"Action failed, providing feedback to model: {last_action_feedback}")
                break 
            time.sleep(1)
        
        if not actions_list:
            last_action_feedback = "No actions were returned. The task is likely not complete. Re-evaluating."
            print(last_action_feedback)
        
        iteration_count += 1
    return synthesized_summary


if __name__ == "__main__":
    repl_loop()