#!/usr/bin/env python3
"""
Hotel Document Generation Script

This script generates 100 hotel descriptions using the LLM and saves them to a CSV file.
It replicates the functionality from the doc_gen.ipynb notebook.
"""

import os
import yaml
import sys
import json
import csv
from pathlib import Path

# Add src directory to Python path
src_path = Path.cwd() / 'src'
sys.path.insert(0, str(src_path))

from dotenv import load_dotenv
from nl_pe.llm import LLM_CLASSES
from nl_pe.llm.prompter import Prompter

def main():
    """Main function to generate hotel documents and save to CSV."""

    print("Starting hotel document generation...")

    # Load configuration
    config_path = Path('configs/llm/config.yaml')
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)

    load_dotenv()

    # Instantiate the Prompter
    prompter = Prompter(config)

    # Prompt with the template
    template_path = 'hotel_gen.jinja2'
    print(f"Prompting LLM with template: {template_path}")
    response = prompter.prompt_from_temp(template_path)

    print("Full response:", response)

    # Parse JSON response
    if 'JSON_dict' in response:
        json_data = response['JSON_dict']

        # Handle both doc_list format and doc_1, doc_2, ... format
        if 'doc_list' in json_data:
            doc_list = json_data['doc_list']
            print(f"Successfully parsed JSON with {len(doc_list)} documents in doc_list format")
        else:
            # Extract docs from doc_1, doc_2, ... format
            doc_list = []
            for i in range(1, 101):  # Assuming 100 documents
                doc_key = f'doc_{i}'
                if doc_key in json_data:
                    doc_list.append(json_data[doc_key])
            print(f"Successfully parsed JSON with {len(doc_list)} documents in doc_1-doc_100 format")

        json_data['doc_list'] = doc_list  # Normalize to doc_list format
    else:
        print("No JSON_dict found in response")
        json_data = None

    # Create CSV file from parsed JSON data
    if json_data and 'doc_list' in json_data:
        # Create data/gen_docs directory if it doesn't exist
        output_dir = Path('data/gen_docs')
        output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = output_dir / 'hotels_1.csv'

        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow(['d_id', 'd_text'])

            # Write data rows
            for i, doc_text in enumerate(json_data['doc_list'], 1):
                writer.writerow([i, doc_text])

        print(f"CSV file created successfully: {csv_path}")
        print(f"Total documents saved: {len(json_data['doc_list'])}")
    else:
        print("No data to save to CSV - either JSON parsing failed or no doc_list found")
        return False

    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Hotel document generation completed successfully!")
    else:
        print("\n❌ Hotel document generation failed!")
        sys.exit(1)
