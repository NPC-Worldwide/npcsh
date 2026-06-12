#!/usr/bin/env python3
"""Generate training data with fixed trace parser and train new adapter."""
import csv, json, os, re
from pathlib import Path

def parse_trace_fixed(trace_str):
    if not trace_str or "---TRACE---" not in trace_str:
        return []
    trace = trace_str.split("---TRACE---", 1)[1]
    pattern = r'(?:^|\s*\|\s*)\[(system|user|tool_call|tool|assistant)\]\s*'
    parts = re.split(pattern, trace)
    segments = []
    for i in range(1, len(parts), 2):
        marker = parts[i]
        content = parts[i+1] if i+1 < len(parts) else ""
        segments.append((marker, content.strip()))
    
    examples = []
    for idx, (marker, content) in enumerate(segments):
        if marker == "user":
            instruction = re.sub(r"User Provided Context:.*", "", content, flags=re.DOTALL).strip()
            if instruction.lower().startswith("continue.") or len(instruction) < 10:
                continue
            for j in range(idx+1, len(segments)):
                if segments[j][0] == "user":
                    break
                if segments[j][0] == "tool_call":
                    tc_text = segments[j][1].strip()
                    m = re.match(r"(\w+)\((\{.*\})\)", tc_text, re.DOTALL)
                    if m:
                        fname = m.group(1)
                        args_raw = m.group(2)
                        try:
                            args = json.loads(args_raw)
                        except json.JSONDecodeError:
                            try:
                                import ast
                                args = ast.literal_eval(args_raw)
                            except:
                                continue
                        if fname == "sh":
                            fname = "shell"
                        elif fname in ("stop", "chat", "delegate", "convene"):
                            continue
                        response = f"<tool_call>\n{json.dumps({'name': fname, 'arguments': args}, ensure_ascii=False)}\n</tool_call>"
                        examples.append((instruction, response))
                    break
    return examples

# Load all traces
X, y = [], []
csv.field_size_limit(10**7)
for csv_file in sorted(Path(os.path.expanduser("~/.npcsh/benchmarks/local")).glob("*.csv")):
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("passed", "").lower() in ("true", "1"):
                for instruction, response in parse_trace_fixed(row.get("output", "")):
                    X.append(f"<|im_start|>user\n{instruction}\n<|im_end|>\n<|im_start|>assistant\n")
                    y.append(f"{response}\n<|im_end|>\n")

print(f"Total examples: {len(X)}")

# Save as JSONL for mlx_lm
output_dir = Path("/tmp/qwen35_v2_data")
output_dir.mkdir(exist_ok=True)
with open(output_dir / "train.jsonl", "w") as f:
    for instruction, response in zip(X, y):
        text = instruction + response
        f.write(json.dumps({"text": text}) + "\n")

print(f"Saved to {output_dir}/train.jsonl")
