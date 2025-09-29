import os
import pandas as pd
import sys
import json
import csv

from pathlib import Path
import yaml
from dotenv import load_dotenv
from nl_pe.llm.prompter import Prompter

CONFIG_PATH = "configs/llm/config.yaml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

load_dotenv()

prompter = Prompter(config)

# Load the user preference CSV
user_csv_path = 'data/users/movielens/movie_users_1.csv'
df = pd.read_csv(user_csv_path)

# Define k_words_max
k_words_max = 3

# Template path
template_path = "pref_aspect_selection.jinja2"  
#'pref_compression.jinja2'

# Process each user preference
comp_prefs = []
for idx, row in df.iterrows():
    user_pref = row['u_pref']

    # Create prompt dict
    prompt_dict = {
        'k_words_max': k_words_max,
        'user_pref': user_pref
    }

    # Call prompter
    response = prompter.prompt_from_temp(template_path, prompt_dict)

    # Parse the response (assuming JSON_dict key)
    if 'JSON_dict' in response:
        json_data = response['JSON_dict']
        comp_pref = json_data.get('output', '')

        # Ensure it's a string
        if isinstance(comp_pref, str):
            comp_prefs.append(comp_pref)
        else:
            comp_prefs.append(str(comp_pref))
        print(f"Processed user {row['u_id']}: compressed to '{comp_pref}'")
    else:
        print(f"No JSON_dict found for user {row['u_id']}")
        comp_prefs.append('')

# Add the comp_pref column
df['comp_pref'] = comp_prefs

# Save back to the same CSV file
df.to_csv(user_csv_path, index=False)

print(f"Added comp_pref column to {user_csv_path}")
