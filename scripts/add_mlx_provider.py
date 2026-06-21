#!/usr/bin/env python3
"""Add native MLX provider support to npcsh."""
import sys

with open('/Users/caug/npcww/npc-core/npcpy/npcpy/gen/response.py', 'r') as f:
    content = f.read()

mlx_func = '''

def get_mlx_response(
    prompt: str = None,
    model: str = None,
    tools: list = None,
    tool_choice: Dict = None,
    tool_map: Dict = None,
    think=None,
    format: Union[str, BaseModel] = None,
    messages: List[Dict[str, str]] = None,
    stream: bool = False,
    attachments: List[str] = None,
    auto_process_tool_calls: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Generate response using mlx_lm directly (no server required).

    Supports both base models and LoRA adapters.
    - If model is a directory with adapter_config.json, loads as adapter
    - Otherwise treats model as mlx-community model ID
    """
    try:
        from mlx_lm import load as mlx_load, generate as mlx_generate
        from mlx_lm.lora import load_adapters
    except ImportError:
        return {
            "response": "",
            "messages": messages or [],
            "error": "mlx-lm not installed. Install with: pip install mlx-lm"
        }

    result = {
        "response": None,
        "messages": messages.copy() if messages else [],
        "raw_response": None,
        "tool_calls": [],
        "tool_results": []
    }

    if prompt:
        if result['messages'] and result['messages'][-1]["role"] == "user":
            result['messages'][-1]["content"] = prompt
        else:
            result['messages'].append({"role": "user", "content": prompt})

    adapter_path = None
    base_model = model
    expanded = os.path.expanduser(model)
    if os.path.isdir(expanded):
        adapter_config_path = os.path.join(expanded, 'adapter_config.json')
        if os.path.exists(adapter_config_path):
            adapter_path = expanded
            with open(adapter_config_path, 'r') as f:
                cfg = json.load(f)
            base_model = cfg.get('model', model)
            print(f"[mlx] Loading adapter: {adapter_path} on base: {base_model}")

    try:
        mlx_model, tokenizer = mlx_load(base_model)
        if adapter_path:
            load_adapters(mlx_model, adapter_path)

        chat_text = ""
        for msg in result["messages"]:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                chat_text += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == "user":
                chat_text += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                chat_text += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        chat_text += "<|im_start|>assistant\n"

        max_tokens = kwargs.get("max_tokens", 512)
        temperature = kwargs.get("temperature", 0.7)
        verbose = kwargs.get("verbose", False)

        response_content = mlx_generate(
            mlx_model,
            tokenizer,
            prompt=chat_text,
            max_tokens=max_tokens,
            temp=temperature,
            verbose=verbose,
        )

        result["response"] = response_content
        result["raw_response"] = response_content
        result["messages"].append({"role": "assistant", "content": response_content})

    except Exception as e:
        logger.error(f"MLX inference error: {e}")
        result["error"] = f"MLX inference error: {str(e)}"
        result["response"] = ""

    return result
'''

insert_point = content.find("def get_llamacpp_response(")
if insert_point == -1:
    print("ERROR: Could not find insert point")
    sys.exit(1)

content = content[:insert_point] + mlx_func + "\n" + content[insert_point:]

dispatch_code = """    elif provider == 'mlx':
        return get_mlx_response(
            prompt=prompt,
            model=model,
            tools=tools,
            tool_choice=tool_choice,
            tool_map=tool_map,
            think=think,
            format=format,
            messages=messages,
            stream=stream,
            attachments=attachments,
            auto_process_tool_calls=auto_process_tool_calls,
            **kwargs
        )
"""

old_omlx = "    elif provider == 'omlx':"
content = content.replace(old_omlx, dispatch_code + old_omlx)

with open('/Users/caug/npcww/npc-core/npcpy/npcpy/gen/response.py', 'w') as f:
    f.write(content)

print("Added get_mlx_response() and 'mlx' provider dispatch")
