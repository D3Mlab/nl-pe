import yaml
import json
import argparse
from dotenv import load_dotenv
import os
from pathlib import Path
import time
from nl_pe.utils.setup_logging import setup_logging
from nl_pe.embedding import EMBEDDER_CLASSES
from nl_pe import search_agent

class ExperimentManager():

    def __init__(self, exp_dir):
        self.exp_dir = exp_dir
        self.load_config()
        self.setup_logger()
        self.config['exp_dir'] = self.exp_dir

    def index_corpus(self):
        self.logger.info("Starting corpus indexing...")

        self.embedding_config = self.config.get('embedding', {})
        self.data_config = self.config.get('data', {})

        #init embedder
        embedder_init_kwargs = dict(
            config = self.config,
            normalize = self.embedding_config.get('normalize', True),
            model_name= self.embedding_config.get('model', ''),
        )

        matryoshka_dim = self.embedding_config.get('matryoshka_dim', None)
        if matryoshka_dim:
            embedder_init_kwargs["matryoshka_dim"] = matryoshka_dim

        embedder_class = EMBEDDER_CLASSES[self.embedding_config.get('class')]
        self.embedder = embedder_class(**embedder_init_kwargs)

        #get index method
        index_method_name = self.embedding_config.get('index_method', '')
        self.index_method = getattr(self.embedder, index_method_name)

        #inputs to index methods:
        # texts_csv_path, index_path, batch_size, prompt

        start_time = time.time()

        self.index_method(
            texts_csv_path = self.data_config.get('d_text_csv', ''),
            index_path = self.data_config.get('index_path', ''),
            batch_size = self.embedding_config.get('batch_size', None),
            prompt = self.embedding_config.get('doc_prompt', '')
        )

        end_time = time.time()
        embedding_time = end_time - start_time

        embedding_details_path = os.path.join(self.exp_dir, "embedding_details.json")
        with open(embedding_details_path, 'w') as f:
            json.dump({'embedding_time': embedding_time}, f)

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
