import os
import pandas as pd
import sys
import json
import csv

from pathlib import Path
import yaml

PROJECT_ROOT = Path.cwd().parent  # points to nl-pe/ (from scripts/)
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))  # make nl_pe importable

os.chdir(PROJECT_ROOT)

from dotenv import load_dotenv
from nl_pe.llm.prompter import Prompter
from nl_pe.utils.text_processing import list_to_text_block

CONFIG_PATH = "configs/llm/config.yaml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

load_dotenv()

prompter = Prompter(config)

# Load the movie corpus and take first 3 rows
corpus_path = 'data/real_docs/movielens/movies_1.csv'
df = pd.read_csv(corpus_path)

# Get first 3 rows and extract d_text column
first_three_texts = df['d_text'].head(3).tolist()

# Convert to formatted text block
item_list = list_to_text_block(first_three_texts, index_str="Movie")

# Create prompt dict
prompt_dict = {'item_list': item_list}

# Use the specified template
template_path = 'movie_pref_gen_1.jinja2'

# Call prompter
response = prompter.prompt_from_temp(template_path, prompt_dict)

print("Full response:", response)

# Parse the response (assuming similar structure to doc_gen)
if 'JSON_dict' in response:
    json_data = response['JSON_dict']

    # Extract preferences from user_1, user_2, ... format
    user_prefs = []
    i = 1
    while True:
        user_key = f'user_{i}'
        if user_key in json_data:
            user_prefs.append(json_data[user_key])
            i += 1
        else:
            break

    print(f"Successfully parsed JSON with {len(user_prefs)} user preferences in user_1-user_{i-1} format")

    # Save to CSV
    output_dir = Path('data/users/movielens')
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / 'movie_users_1.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['u_id', 'u_pref'])
        for idx, pref_text in enumerate(user_prefs, 1):
            writer.writerow([idx, pref_text])

    print(f"CSV file created successfully: {csv_path}")
else:
    print("No JSON_dict found in response")
