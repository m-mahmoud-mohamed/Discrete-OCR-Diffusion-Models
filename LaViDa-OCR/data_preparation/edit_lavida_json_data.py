import json
import re
from tqdm import tqdm

# Files
input_file = "/path/to/olmocr_train.json"   # Input JSON from convert_olmocr_parallel.py
output_file = "/path/to/olmocr_train_final.json"  # Cleaned output for training

PROMPT = """<image>
Extract all text from this page in Markdown format. Follow these rules:
1. Preserve all headings, lists, tables, and formulas.
2. Use LaTeX for math equations (e.g., $x^2$).
3. Do not describe images; only transcribe text inside them.
4. Maintain the original reading order."""


yaml_pattern = re.compile(r"^---\n.*?\n---\n", re.DOTALL)

print("Loading data...")
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Processing {len(data)} samples...")

cleaned_count = 0
for item in tqdm(data):
    conversations = item.get('conversations', [])
    
    # Update HUMAN prompt and GPT target
    for turn in conversations:
        # 1. Update Prompt
        if turn['from'] == 'human':
            turn['value'] = PROMPT
            
        # 2. Clean Target (Remove YAML)
        elif turn['from'] == 'gpt':
            original_text = turn['value']
            # Strip the --- ... --- header
            cleaned_text = yaml_pattern.sub("", original_text).strip()
            turn['value'] = cleaned_text
            cleaned_count += 1

print(f"Saving to {output_file}...")
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

