import os
import yaml
import argparse
import re


def modify_measures_in_config(config, k):
    if 'measures' in config and isinstance(config['measures'], list):
        # Collect unique prefixes that have numeric suffixes
        prefixes = set()
        non_numeric_measures = []
        for measure in config['measures']:
            if re.match(r'^(.+?)_(\d+)$', measure):
                prefix = measure.rsplit('_', 1)[0]
                prefixes.add(prefix)
            else:
                non_numeric_measures.append(measure)
        # Generate new measures for each prefix from 1 to k
        new_measures = []
        for prefix in prefixes:
            for i in range(1, k + 1):
                new_measures.append(f"{prefix}_{i}")
        # Add any non-numeric measures back
        new_measures.extend(non_numeric_measures)
        config['measures'] = new_measures
    return config


def process_directory(parent_dir, k):
    for root, dirs, files in os.walk(parent_dir):
        if 'eval_config.yaml' in files:
            config_path = os.path.join(root, 'eval_config.yaml')
            print(f"Processing {config_path}")
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            if config is None:
                config = {}
            config = modify_measures_in_config(config, k)
            with open(config_path, 'w') as f:
                yaml.safe_dump(config, f, default_flow_style=None)
            print(f"Modified measures in {config_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modify eval_config.yaml measures cutoff values to a specified k.")
    parser.add_argument("parent_dir", type=str, help="Parent directory to process recursively.")
    parser.add_argument("--k", type=int, default=1000, help="New cutoff value for measures (default: 1000).")
    args = parser.parse_args()

    if not os.path.isdir(args.parent_dir):
        parser.error(f"Directory {args.parent_dir} does not exist.")

    process_directory(args.parent_dir, args.k)
    print(f"Completed modifying measures cutoff values to {args.k} in eval_config.yaml files under {args.parent_dir}")
