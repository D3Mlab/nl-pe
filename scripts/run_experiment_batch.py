import os
import argparse
import subprocess
import sys
from dotenv import load_dotenv

def run_experiment_batch(batch_dir, exp_type, skip_existing=False):
    # Load environment variables
    load_dotenv()

    # Find all config.yaml files in the directory and subdirectories
    exp_dirs = []
    for root, _, files in os.walk(batch_dir):
        if "config.yaml" in files:
            exp_dirs.append(root)

    # Run each experiment using subprocess with new cmd line interface
    for exp_dir in exp_dirs:
        print(f"Running {exp_type} in directory: {exp_dir}")
        cmd = [
            sys.executable,
            'src/nl_pe/experiment_manager.py',
            '-c', exp_dir,
            '-e', exp_type,
        ]
        if skip_existing:
            cmd.append('-se')

        result = subprocess.run(cmd, cwd='.')
        if result.returncode != 0:
            print(f"FAILED: {exp_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a batch of experiments in the specified directory.")
    parser.add_argument("-c", "--batch-dir", type=str, required=True, help="The path to the directory containing the batch of experiments.")
    parser.add_argument("-e", "--exp-type", type=str, required=True, help="The type of experiment to run (e.g., index_corpus, ir_exp)")
    parser.add_argument("-se", "--skip-existing", action="store_true", help="Forward skip-existing flag to experiment_manager.py")
    args = parser.parse_args()

    run_experiment_batch(args.batch_dir, args.exp_type, skip_existing=args.skip_existing)
