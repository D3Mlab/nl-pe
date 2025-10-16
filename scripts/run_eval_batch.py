import os
import argparse
import subprocess
import sys
from dotenv import load_dotenv

def run_eval_batch(e, skip_existing):
    # Load environment variables
    load_dotenv()

    # Find all eval_config.yaml files in the directory and subdirectories
    exp_dirs = []
    for root, _, files in os.walk(e):
        if "eval_config.yaml" in files:
            exp_dirs.append(root)

    # Run evaluation for each experiment using subprocess with new cmd line interface
    for exp_dir in exp_dirs:
        print(f"Evaluating experiment in directory: {exp_dir}")
        cmd = [sys.executable, 'src/nl_pe/eval_manager.py', '-c', exp_dir]
        if skip_existing:
            cmd.append('--skip-existing')
        subprocess.run(cmd, cwd='.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation for a batch of experiments in the specified directory.")
    parser.add_argument("-c", "--eval-dir", type=str, required=True, help="The path to the directory containing the batch of evaluations.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip evaluation if output files already exist")
    args = parser.parse_args()

    run_eval_batch(args.eval_dir, args.skip_existing)
