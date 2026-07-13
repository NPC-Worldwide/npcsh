#!/usr/bin/env python3
"""Append curated benchmark tasks to tasks.csv."""

import pandas as pd

TASKS = [
    {
        "id": "shell-11",
        "category": "shell",
        "difficulty": "medium",
        "setup_cmd": "mkdir -p /tmp/exttest && touch /tmp/exttest/a.py /tmp/exttest/b.py /tmp/exttest/c.txt /tmp/exttest/d.txt /tmp/exttest/e.sh",
        "instruction": "Write a bash script /tmp/count_ext.sh that takes a directory path as argument and prints the count of files for each file extension found, one per line in the format 'ext: count'. Then run it on /tmp/exttest.",
        "verify_cmd": "test -f /tmp/count_ext.sh && bash /tmp/count_ext.sh /tmp/exttest && bash -c 'out=$(bash /tmp/count_ext.sh /tmp/exttest); echo \"$out\" | grep -q \"py:\" && echo \"$out\" | grep -q \"txt:\"'",
        "description": "Bash script to count files by extension",
    },
    {
        "id": "file-ops-11",
        "category": "file-ops",
        "difficulty": "easy",
        "setup_cmd": "",
        "instruction": "Create /tmp/notes.txt with 10 lines of text (any content). Then create /tmp/notes_sorted.txt containing the same lines sorted alphabetically.",
        "verify_cmd": "test -f /tmp/notes.txt && test -f /tmp/notes_sorted.txt && test $(wc -l < /tmp/notes.txt) -eq 10 && test $(wc -l < /tmp/notes_sorted.txt) -eq 10",
        "description": "Create file and sort its contents",
    },
    {
        "id": "python-11",
        "category": "python",
        "difficulty": "medium",
        "setup_cmd": "",
        "instruction": "Write /tmp/fizzbuzz.py that prints numbers 1 to 30, but prints 'Fizz' for multiples of 3, 'Buzz' for multiples of 5, and 'FizzBuzz' for multiples of both. Run it and verify the output contains 'FizzBuzz'.",
        "verify_cmd": "test -f /tmp/fizzbuzz.py && python3 /tmp/fizzbuzz.py | grep -q 'FizzBuzz'",
        "description": "Classic FizzBuzz in Python",
    },
    {
        "id": "python-12",
        "category": "python",
        "difficulty": "medium",
        "setup_cmd": "",
        "instruction": "Write a Python script /tmp/fib.py that defines a function fibonacci(n) returning the nth fibonacci number. Then print fibonacci(10).",
        "verify_cmd": "test -f /tmp/fib.py && python3 /tmp/fib.py | grep -q '55'",
        "description": "Fibonacci function in Python",
    },
    {
        "id": "data-11",
        "category": "data",
        "difficulty": "medium",
        "setup_cmd": "",
        "instruction": "Create /tmp/weather.csv with columns city,temperature,humidity for 5 cities. Then write /tmp/avg_temp.py that reads the CSV, computes the average temperature, and prints it.",
        "verify_cmd": "test -f /tmp/weather.csv && test -f /tmp/avg_temp.py && python3 /tmp/avg_temp.py",
        "description": "CSV creation and average computation",
    },
    {
        "id": "text-11",
        "category": "text",
        "difficulty": "easy",
        "setup_cmd": "",
        "instruction": "Create /tmp/words.txt with 5 lines of mixed-case words. Then write /tmp/words_upper.txt containing the same words converted to uppercase.",
        "verify_cmd": "test -f /tmp/words.txt && test -f /tmp/words_upper.txt && test $(wc -l < /tmp/words.txt) -eq 5 && test $(wc -l < /tmp/words_upper.txt) -eq 5",
        "description": "Text case conversion",
    },
    {
        "id": "system-11",
        "category": "system",
        "difficulty": "easy",
        "setup_cmd": "",
        "instruction": "List all users on the system (from /etc/passwd) and write just the usernames to /tmp/usernames.txt, one per line.",
        "verify_cmd": "test -f /tmp/usernames.txt && test $(wc -l < /tmp/usernames.txt) -ge 5 && head -1 /tmp/usernames.txt | grep -q '^[a-z]'",
        "description": "Extract system usernames",
    },
    {
        "id": "debug-11",
        "category": "debug",
        "difficulty": "medium",
        "setup_cmd": "cat > /tmp/fix_loop.py <<'EOF'\nnumbers = [1, 2, 3, 4, 5]\ntotal = 0\nfor i in range(len(numbers)):\n    total = total + i\nprint(total)\nEOF",
        "instruction": "Fix the bug in /tmp/fix_loop.py. It should sum the values in the list, not the indices. The correct output should be 15. Overwrite the file with the fixed code.",
        "verify_cmd": "test -f /tmp/fix_loop.py && python3 /tmp/fix_loop.py | grep -q '15'",
        "description": "Fix off-by-one loop bug",
    },
    {
        "id": "shell-12",
        "category": "shell",
        "difficulty": "medium",
        "setup_cmd": "",
        "instruction": "Write a bash script /tmp/fileinfo.sh that takes a file path as argument and prints: its size in bytes, number of lines, and last modification timestamp (in seconds since epoch), separated by spaces. Test it on /etc/passwd.",
        "verify_cmd": "test -f /tmp/fileinfo.sh && bash /tmp/fileinfo.sh /etc/passwd | grep -qE '[0-9]+ [0-9]+ [0-9]+'",
        "description": "Bash script for file metadata",
    },
    {
        "id": "python-13",
        "category": "python",
        "difficulty": "medium",
        "setup_cmd": "",
        "instruction": "Write /tmp/matrix.py that multiplies two 2x2 matrices A=[[1,2],[3,4]] and B=[[5,6],[7,8]] and prints the resulting 2x2 matrix, one row per line with space-separated values.",
        "verify_cmd": "test -f /tmp/matrix.py && python3 /tmp/matrix.py | grep -q '19 22'",
        "description": "2x2 matrix multiplication in Python",
    },
]


def main():
    path = "/Users/caug/npcww/npc-core/npcsh/npcsh/benchmark/tasks.csv"
    df = pd.DataFrame(TASKS, columns=["id", "category", "difficulty", "setup_cmd", "instruction", "verify_cmd", "description"])
    df.to_csv(path, mode="a", index=False, header=False)
    print(f"Appended {len(TASKS)} tasks to {path}")
    existing = pd.read_csv(path)
    print(f"Total tasks: {len(existing)}")


if __name__ == "__main__":
    main()
