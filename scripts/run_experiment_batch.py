import os
import argparse
from nl_pe.experiment_manager import ExperimentManager
from dotenv import load_dotenv

def run_experiment_batch(batch_dir):
    # Load environment variables
    load_dotenv()

    # Find all config.yaml files in the directory and subdirectories
    exp_dirs = []
    for root, _, files in os.walk(batch_dir):
        if "config.yaml" in files:
            exp_dirs.append(root)

    # Run each experiment
    for exp_dir in exp_dirs:
        print(f"Running experiment in directory: {exp_dir}")
        manager = ExperimentManager(exp_dir)
        manager.run_experiment()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a batch of experiments in the specified directory.")
    parser.add_argument("-e", type=str, help="The path to the directory containing the batch of experiments.")
    args = parser.parse_args()

    run_experiment_batch(args.e)