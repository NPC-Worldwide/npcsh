from npcpy.data.image import capture_screenshot
import time
import os 
import platform
from npcpy.llm_funcs import get_llm_response
from npcpy.work.desktop import perform_action
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import imagehash # Using perceptual hashing for smarter screen comparison
from npcsh._state import NPCSH_VISION_MODEL, NPCSH_VISION_PROVIDER
import argparse
from npcpy.npc_compiler import NPC

def get_system_examples():
    system = platform.system()
    if system == "Windows":
        return "Examples: start firefox, notepad, calc, explorer"
    elif system == "Darwin":
        return "Examples: open -a Firefox, open -a TextEdit, open -a Calculator"
    else:
        return "Examples: firefox &, gedit &, gnome-calculator &"

def format_plonk_summary(synthesized_summary: list) -> str:
    """Formats the summary of a plonk session into a readable markdown report."""
    if not synthesized_summary:
        return "Plonk session ended with no actions performed."

    output = "## Plonk Session Summary\n\n"
    for info in synthesized_summary:
        iteration = info.get('iteration', 'N/A')
        feedback = info.get('last_action_feedback', 'None')
        coords = info.get('last_click_coords', 'None')
        output += f"### Iteration {iteration}\n"
        output += f"- **Feedback:** {feedback}\n"
        output += f"- **Last Click:** {coords}\n\n"
    return output

def get_image_hash(image_path):
    """Generate a perceptual hash of the image to detect screen changes intelligently."""
    try:
        # Perceptual hash is more robust to minor changes like a blinking cursor
        return imagehash.phash(Image.open(image_path))
    except Exception as e:
        print(f"Could not generate image hash: {e}")
        return None

def add_click_vector_trail(image_path, click_history, output_path):
    """Add click markers showing the progression/trail of clicks with arrows and numbers."""
    try:
        img = Image.open(image_path)
        img_array = np.array(img)
        height, width = img_array.shape[:2]
        
        fig, ax = plt.subplots(1, 1, figsize=(width/100, height/100), dpi=100)
        ax.imshow(img_array)
        
        font_size = max(12, min(width, height) // 80)
        colors = plt.cm.viridis(np.linspace(0.3, 1.0, len(click_history)))
        
        # Draw arrows connecting clicks first
        if len(click_history) > 1:
            for i in range(len(click_history) - 1):
                x1, y1 = (click_history[i]['x'] * width / 100, click_history[i]['y'] * height / 100)
                x2, y2 = (click_history[i+1]['x'] * width / 100, click_history[i+1]['y'] * height / 100)
                ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                           arrowprops=dict(arrowstyle='->,head_width=0.6,head_length=0.8', 
                                         lw=3, color='cyan', alpha=0.9, shrinkA=25, shrinkB=25))

        # Draw numbered points and coordinate labels for ALL points
        for i, click in enumerate(click_history):
            x_pixel = int(click['x'] * width / 100)
            y_pixel = int(click['y'] * height / 100)
            
            radius = 25
            circle = patches.Circle((x_pixel, y_pixel), radius=radius,
                                   linewidth=3, edgecolor='white', 
                                   facecolor=colors[i], alpha=0.9)
            ax.add_patch(circle)
            
            # Draw the number inside the circle
            ax.text(x_pixel, y_pixel, str(i+1),
                    fontsize=font_size + 4, 
                    color='white', weight='bold', ha='center', va='center')
            
            # FIXED: Draw the coordinate text label for EVERY point
            coord_text = f"({click['x']}, {click['y']})"
            ax.text(x_pixel + radius + 5,  # Position text to the right of the circle
                    y_pixel,              # Vertically centered with the circle
                    coord_text,
                    fontsize=font_size, 
                    color='white',
                    weight='bold',
                    ha='left', va='center',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=colors[i], 
                              alpha=0.9, edgecolor='white'))
        
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close()
        
        return True
    except Exception as e:
        print(f"Failed to add click trail with matplotlib: {e}")
        return False

def execute_plonk_command(request, model, provider, npc=None, plonk_context=None, max_iterations=10, debug=False):
    system_examples = get_system_examples()
    messages = []
    last_action_feedback = "None"
    last_click_coords = None
    synthesized_summary = []
    
    current_screen_hash = None
    click_history = []
    HASH_DISTANCE_THRESHOLD = 3

    for iteration_count in range(max_iterations):
        try: 
            screenshot_info = capture_screenshot(full=True)
            screenshot_path = screenshot_info.get('file_path') if screenshot_info else None
            
            if not screenshot_path:
                last_action_feedback = "Error: Failed to capture screenshot."
                time.sleep(1)
                continue

            new_screen_hash = get_image_hash(screenshot_path)
            
            if current_screen_hash is None or (new_screen_hash - current_screen_hash > HASH_DISTANCE_THRESHOLD):
                if debug and current_screen_hash is not None:
                    print(f"Screen changed (hash distance: {new_screen_hash - current_screen_hash}) - resetting click history.")
                click_history = []
                current_screen_hash = new_screen_hash
            
            summary_info = {
                'iteration': iteration_count + 1,
                'last_action_feedback': last_action_feedback,
                'last_click_coords': click_history[-1] if click_history else None
            }
            synthesized_summary.append(summary_info)

            if debug:
                print(f"Iteration {iteration_count + 1}/{max_iterations}")
            
            context_injection = ""
            if plonk_context:
                context_injection = f"""
            ---
            IMPORTANT TEAM CONTEXT FOR THIS TASK:
            {plonk_context}
            ---
            """
            
            completion_example_text = """
            {
              "actions": [],
              "status": "Task appears complete. Waiting for user approval to proceed or finish."
            }
            """
            
            quit_rule_text = 'NEVER include {"type": "quit"} in your actions - the user controls when to stop.'

            prompt_examples = """
            ---
            EXAMPLE 1: Task "Create and save a file named 'memo.txt' with the text 'Meeting at 3pm'"
            {
              "actions": [
                { "type": "bash", "command": "gedit &" },

                {"type":"click", "x": 10, "y": 30}
              ]
            }
            ---
            EXAMPLE 2: Task "Search for news about space exploration"
            {
              "actions": [
                { "type": "bash", "command": "firefox &" },

                          ]
            }
            ---
            EXAMPLE 3: Task "Click the red button on the form"
            {
              "actions": [
                { "type": "click", "x": 75, "y": 45 }
              ]
            }
            ---
            EXAMPLE 4: Task "Open Gmail and draft a reply to most recent email"
            {
              "actions": [
                { "type": "bash", "command": "open -a Safari" },

                              ]
            }
            """
            
            prompt_template = f"""
            Goal: {request}
            Feedback from last action: {last_action_feedback}

           {context_injection}

            Your task is to control the computer to achieve the goal.
            
            IMPORTANT: You should take actions step-by-step and verify each step works before proceeding.
            DO NOT plan all actions at once - take a few actions, then look at the screen again.
            
            CRITICAL: NEVER use the 'quit' action automatically. Even if the task appears complete,
            continue working or wait for user guidance. The user will decide when to quit.
            
            THOUGHT PROCESS:
            1. Analyze the screen. Is the application I need (e.g., a web browser) already open?
            2. If YES, `click` it. If NO, use `bash` to launch it. Use the examples: {system_examples}.
            3. Take 2-3 actions maximum, then let me see the screen again to verify progress.
            4. If task appears complete, explain status but DO NOT quit - wait for user direction.
            
            Your response MUST be a JSON object with an "actions" key.
            All clicking actions should use percentage coordinates relative 
            to the screen size. 
            The x and y are (0,0) at the TOP LEFT CORNER OF THE SCREEN.
            
            MAXIMUM 3 ACTIONS PER RESPONSE - then let me see the screen to verify progress.
            Never do more than one click, type, or hotkey event per response. It is important to take a sequence of 
            slow actions separated to avoid making mistakes and falling in loops.
            
            If the task appears complete, you can include an empty actions list and explain:
            {completion_example_text}
            
            {quit_rule_text}
            """ + prompt_examples

            image_to_send_path = screenshot_path
            
            if click_history:
                marked_image_path = "/tmp/marked_screenshot.png"
                if add_click_vector_trail(screenshot_path, click_history, marked_image_path):
                    image_to_send_path = marked_image_path
                    if debug:
                        print(f"Drew click trail with {len(click_history)} points.")

            response = get_llm_response(prompt_template, model=model, provider=provider, npc=npc, 
                                        images=[image_to_send_path], messages=messages, format="json")
            messages = response.get("messages", messages)
            response_data = response.get('response')
            
            if debug:
                print(response_data)

            if not isinstance(response_data, dict) or "actions" not in response_data:
                last_action_feedback = f"Invalid JSON response from model: {response_data}"
                continue

            actions_list = response_data.get("actions", [])
            if not isinstance(actions_list, list):
                last_action_feedback = "Model did not return a list in the 'actions' key."
                continue
            
            for action in actions_list:
                if debug:
                    print(f"Executing action: {action}")
                
                if action.get("type") == "quit":
                    print("⚠️  Model attempted to quit automatically. Ignoring.")
                    continue
                    
                result = perform_action(action)
                last_action_feedback = result.get("message") or result.get("output")

                if action.get("type") == "click":
                    click_info = {"x": action.get("x"), "y": action.get("y")}
                    click_history.append(click_info)
                    if len(click_history) > 6:
                        click_history.pop(0)
                
                if result.get("status") == "error":
                    last_action_feedback = f"Action failed: {last_action_feedback}"
                    print(f"Action failed, providing feedback to model: {last_action_feedback}")
                    break 
                time.sleep(1)

            if response_data.get("status") and "complete" in response_data.get("status", "").lower():
                print(f"🎯 Model reports: {response_data.get('status')}")
                print("   Press Ctrl+C to provide guidance or approval, or let it continue...")

            if not actions_list:
                last_action_feedback = "No actions were returned by the model. Re-evaluating."
                if debug:
                    print(last_action_feedback)
        
        except KeyboardInterrupt:
            print("\n⚠️ Plonk paused. Provide additional guidance or press Enter to continue.")
            try:
                user_guidance = input("Guidance > ").strip()
                if user_guidance:
                    request += f"\n\n---\nUser Guidance: {user_guidance}\n---"
                    last_action_feedback = "User provided new guidance to correct the course."
                    print("✅ Guidance received. Resuming with updated instructions...")
                else:
                    last_action_feedback = "User paused and resumed without new guidance."
                    print("✅ No guidance provided. Resuming...")
                continue
            except EOFError:
                print("\nExiting plonk mode.")
                break 

    return synthesized_summary

def main():
    parser = argparse.ArgumentParser(description="Execute GUI automation tasks using vision models")
    parser.add_argument("request", help="The task to perform")
    parser.add_argument("--model", help="Model to use")
    parser.add_argument("--provider", help="Provider to use") 
    parser.add_argument("--max-iterations", type=int, default=10, help="Maximum iterations")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--npc", type=str, default=os.path.expanduser('~/.npcsh/npc_team/plonk.npc'), help="Path to NPC file")
    
    args = parser.parse_args()
    
    npc = NPC(file=args.npc) if os.path.exists(os.path.expanduser(args.npc)) else None
    
    model = args.model or (npc.model if npc else NPCSH_VISION_MODEL)
    provider = args.provider or (npc.provider if npc else NPCSH_VISION_PROVIDER)

    summary = execute_plonk_command(
        request=args.request,
        model=model,
        provider=provider,
        npc=npc,
        max_iterations=args.max_iterations,
        debug=args.debug
    )
    
    print(format_plonk_summary(summary))

if __name__ == "__main__":
    main()