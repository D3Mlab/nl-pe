import argparse
import shutil
from pathlib import Path


def should_skip_dir(path: Path) -> bool:
    """Return True if this path is inside a per_query_results tree."""
    return "per_query_results" in path.parts


def main():
    parser = argparse.ArgumentParser(
        description="Copy eval_config.yaml into all subdirs containing config.yaml (excluding per_query_results)."
    )
    parser.add_argument("-s", "--source", required=True, help="Path to source eval_config.yaml")
    parser.add_argument("-d", "--dest", required=True, help="Destination root directory to walk")

    args = parser.parse_args()

    src = Path(args.source)
    dest_root = Path(args.dest)

    if not src.exists():
        raise FileNotFoundError(f"Source eval config not found: {src}")

    if not dest_root.exists():
        raise FileNotFoundError(f"Destination root not found: {dest_root}")

    copied = 0
    skipped = 0

    for path in dest_root.rglob("*"):
        if not path.is_dir():
            continue

        if should_skip_dir(path):
            continue

        config_file = path / "config.yaml"
        if config_file.exists():
            target = path / "eval_config.yaml"
            shutil.copy2(src, target)
            print(f"[COPY] {target}")
            copied += 1
        else:
            skipped += 1

    print("\nDone.")
    print(f"Copied eval_config.yaml to {copied} directories.")
    print(f"Skipped {skipped} directories without config.yaml.")


if __name__ == "__main__":
    main()
