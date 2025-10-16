import os
import argparse
import subprocess
import sys
from dotenv import load_dotenv

def run_experiment_batch(batch_dir, exp_type):
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
        subprocess.run([sys.executable, 'src/nl_pe/experiment_manager.py', '-c', exp_dir, '-e', exp_type], cwd='.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a batch of experiments in the specified directory.")
    parser.add_argument("-c", "--batch-dir", type=str, required=True, help="The path to the directory containing the batch of experiments.")
    parser.add_argument("-e", "--exp-type", type=str, required=True, help="The type of experiment to run (e.g., index_corpus, ir_exp)")
    args = parser.parse_args()

    run_experiment_batch(args.batch_dir, args.exp_type)
