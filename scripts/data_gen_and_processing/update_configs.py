import os
import sys
import yaml
import copy

UPDATE_DICT = {'logging': {'level': 'INFO'}}

def deep_update(orig, updates):
    """
    Recursively update dict `orig` with values from `updates`.
    """
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(orig.get(k), dict):
            deep_update(orig[k], v)
        else:
            orig[k] = v

def process_config(path):
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}

    original = copy.deepcopy(data)
    deep_update(data, UPDATE_DICT)

    if data != original:
        with open(path, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)
        print(f"Updated: {path}")
    else:
        print(f"No change: {path}")

def walk_and_update(root_dir):
    for root, _, files in os.walk(root_dir):
        if "config.yaml" in files:
            process_config(os.path.join(root, "config.yaml"))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python update_configs.py <root_dir>")
        sys.exit(1)

    root_dir = sys.argv[1]

    if not os.path.isdir(root_dir):
        print(f"Error: '{root_dir}' is not a directory")
        sys.exit(1)

    walk_and_update(root_dir)
