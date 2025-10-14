import yaml
import json
import argparse
from dotenv import load_dotenv
import os
from pathlib import Path
import time
from nl_pe.utils.setup_logging import setup_logging
from nl_pe import search_agent

class ExperimentManager():

    def __init__(self, exp_dir):
        self.exp_dir = exp_dir
        self.load_config()
        #TODO: update configs to direct subclass loggers to experiment log
        self.setup_logger()

    def index_corpus(self):
        self.logger.info("Starting corpus indexing...")        


    def load_config(self):
        config_path = os.path.join(self.exp_dir, "config.yaml")
        with open(config_path, "r") as config_file:
            self.config = yaml.safe_load(config_file)

    def setup_logger(self):
        self.logger = setup_logging(self.__class__.__name__, self.config, output_file=os.path.join(self.exp_dir, "experiment.log"))

    def run_experiment(self, exp_type):
        # Call the method dynamically
        if not hasattr(self, exp_type):
            raise ValueError(f"Experiment type '{exp_type}' is not defined in ExperimentManager.")
        method = getattr(self, exp_type)
        method()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments in the specified directory.")
    parser.add_argument("-d", "--exp-dir", type=str, required=True, help="Path to the experiment directory containing config.yaml")
    parser.add_argument("-e", "--exp-type", type=str, required=True, help="Name of the experiment method to run (e.g., index_corpus)")
    args = parser.parse_args()

    load_dotenv()

    config_path = os.path.join(args.exp_dir, "config.yaml")
    if not os.path.exists(config_path):
        print(f"No config.yaml found in {args.exp_dir}. Skipping experiment.")
    else:
        manager = ExperimentManager(args.exp_dir)
        manager.run_experiment(args.exp_type)
