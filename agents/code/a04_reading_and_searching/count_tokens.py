"""How big is demo_project/? Lines, characters and Qwen3 tokens per file (needs transformers; downloads the tokenizer only).

    pip install transformers && python3 count_tokens.py
"""
from pathlib import Path

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
root = Path(__file__).parent / "demo_project"
files = sorted(root.rglob("*.py"))
total_lines = total_tokens = 0
for f in files:
    text = f.read_text()
    n = len(tok(text)["input_ids"])
    total_lines += text.count("\n")
    total_tokens += n
    print(f"{str(f.relative_to(root)):34} {text.count(chr(10)):5} lines {len(text):7,} chars {n:6,} tokens")
print(f"{len(files)} files, {total_lines:,} lines, {total_tokens:,} tokens")
