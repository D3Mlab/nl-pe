import os
import sys

def delete_log_files(root_dir):
    deleted = 0
    for root, _, files in os.walk(root_dir):
        for name in files:
            if name.endswith(".log"):
                path = os.path.join(root, name)
                try:
                    os.remove(path)
                    print(f"Deleted: {path}")
                    deleted += 1
                except OSError as e:
                    print(f"Failed to delete {path}: {e}")

    print(f"\nTotal .log files deleted: {deleted}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python delete_logs.py <root_dir>")
        sys.exit(1)

    root_dir = sys.argv[1]

    if not os.path.isdir(root_dir):
        print(f"Error: '{root_dir}' is not a directory")
        sys.exit(1)

    delete_log_files(root_dir)
