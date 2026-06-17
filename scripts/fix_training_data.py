#!/usr/bin/env python3
"""Fixed trace parser using regex-based marker splitting."""
import csv
import json
import re

def parse_trace_fixed(trace_str):
    """Parse npcsh trace into (instruction, tool_call) pairs."""
    if not trace_str or "---TRACE---" not in trace_str:
        return []
    
    trace = trace_str.split("---TRACE---", 1)[1]
    
    # Split trace into segments by markers
    pattern = r'(?:^|\s*\|\s*)\[(system|user|tool_call|tool|assistant)\]\s*'
    parts = re.split(pattern, trace)
    
    segments = []
    for i in range(1, len(parts), 2):
        marker = parts[i]
        content = parts[i+1] if i+1 < len(parts) else ""
        segments.append((marker, content.strip()))
    
    print(f"  Segments: {len(segments)}")
    for seg in segments:
        print(f"    {seg[0]}: {seg[1][:80]}")
    
    examples = []
    for idx, (marker, content) in enumerate(segments):
        if marker == "user":
            instruction = content
            # Remove "User Provided Context" if present
            instruction = re.sub(r"User Provided Context:.*", "", instruction, flags=re.DOTALL).strip()
            
            # Skip retry prompts
            if instruction.lower().startswith("continue.") or instruction.lower().startswith("call stop"):
                continue
            if len(instruction) < 10:
                continue
            
            print(f"  Processing user instruction: {instruction[:60]}")
            
            # Find first tool_call after this user message
            for j in range(idx+1, len(segments)):
                if segments[j][0] == "user":
                    break
                if segments[j][0] == "tool_call":
                    tc_text = segments[j][1].strip()
                    print(f"    Found tool_call: {tc_text[:80]}")
                    m = re.match(r"(\w+)\((\{.*\})\)", tc_text, re.DOTALL)
                    if m:
                        fname = m.group(1)
                        args_raw = m.group(2)
                        print(f"    Parsed: fname={fname}")
                        try:
                            args = json.loads(args_raw)
                        except json.JSONDecodeError:
                            try:
                                import ast
                                args = ast.literal_eval(args_raw)
                            except:
                                continue
                        
                        # Normalize
                        if fname == "sh":
                            fname = "shell"
                        elif fname in ("stop", "chat", "delegate", "convene"):
                            print(f"    Skipping {fname}")
                            continue
                        
                        response = f"<tool_call>\n{json.dumps({'name': fname, 'arguments': args}, ensure_ascii=False)}\n</tool_call>"
                        examples.append({"instruction": instruction, "response": response})
                        print("    Added example")
                    break
    
    return examples

if __name__ == "__main__":
    # Test on single trace
    with open("/Users/caug/.npcsh/benchmarks/local/npcsh_ollama_qwen3.5_4b_20260531_223508.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["task_id"] == "shell-01":
                examples = parse_trace_fixed(row.get("output", ""))
                print(f"\nFinal examples from shell-01: {len(examples)}")
                for ex in examples:
                    print(f"\n  INSTRUCTION: {ex['instruction'][:80]}")
                    print(f"  RESPONSE: {ex['response'][:120]}")
                break
