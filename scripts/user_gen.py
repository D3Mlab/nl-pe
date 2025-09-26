import os
import pandas as pd
import sys
import json
import csv

from pathlib import Path
import yaml
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

# Get first n_rows rows and extract d_text column
n_items = 30
first_three_texts = df['d_text'].head(n_items).tolist()

# Convert to formatted text block
item_list = list_to_text_block(first_three_texts, index_str="Movie")
n_users = 5

#word requirements for each preferences
K_words_min = 20
K_words_max = 100

#number of minimum relevant items 
n_rel_min = 2

# Create prompt dict
prompt_dict = {
               'item_list': item_list,
               'n_users': n_users,
               'n_items': n_items,
               'K_words_min': K_words_min,
               'K_words_max': K_words_max,
               'n_rel_min': n_rel_min}

# Use the specified template
template_path = 'item_pref_gen_1.jinja2'

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
        for idx, user_dict in enumerate(user_prefs, 1):
            writer.writerow([idx, user_dict['preference']])

    print(f"CSV file created successfully: {csv_path}")

    # Save to qrels file
    qrels_path = output_dir / 'movie_users_qrels.txt'
    with open(qrels_path, 'w', encoding='utf-8') as f:
        for user_id, user_dict in enumerate(user_prefs, 1):
            for item_id in user_dict['relevant_item_id_list']:
                f.write(f"{user_id} 0 {item_id} 1\n")

    print(f"Qrels file created successfully: {qrels_path}")
else:
    print("No JSON_dict found in response")
