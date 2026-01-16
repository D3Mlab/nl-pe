import os
import sys
import json

# Hardcoded keys to remove from state
KEYS_TO_RMV = [
    "query_emb",
    "doc_ids",
    "posterior_means",
    "posterior_variances",
]


def clean_detailed_results(root_dir):
    modified = 0
    skipped = 0
    errors = 0

    for root, _, files in os.walk(root_dir):
        for name in files:
            if name == "detailed_results.json":
                path = os.path.join(root, name)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    if not isinstance(data, dict):
                        print(f"Skipped (not a dict): {path}")
                        skipped += 1
                        continue

                    removed_any = False
                    for k in KEYS_TO_RMV:
                        if k in data:
                            del data[k]
                            removed_any = True

                    if removed_any:
                        with open(path, "w", encoding="utf-8") as f:
                            json.dump(data, f, indent=2)
                        print(f"Cleaned: {path}")
                        modified += 1
                    else:
                        skipped += 1

                except Exception as e:
                    print(f"Error processing {path}: {e}")
                    errors += 1

    print("\nSummary:")
    print(f"  Files modified: {modified}")
    print(f"  Files skipped : {skipped}")
    print(f"  Errors        : {errors}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python clean_detailed_results.py <root_dir>")
        sys.exit(1)

    root_dir = sys.argv[1]

    if not os.path.isdir(root_dir):
        print(f"Error: '{root_dir}' is not a directory")
        sys.exit(1)

    clean_detailed_results(root_dir)
