import yaml
import json
import argparse
from dotenv import load_dotenv
import os
from pathlib import Path
import time
import pandas as pd
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

        embedder_class = EMBEDDER_CLASSES[self.embedding_config.get('class')]
        self.embedder = embedder_class(self.config)

        #get index method
        index_method_name = self.embedding_config.get('index_method', '')
        self.index_method = getattr(self.embedder, index_method_name)

        start_time = time.time()

        self.index_method(
            texts_csv_path = self.data_config.get('d_text_csv', ''),
            index_path = self.data_config.get('index_path', ''),
            inference_batch_size = self.embedding_config.get('inference_batch_size', None),
            prompt = self.embedding_config.get('doc_prompt', '')
        )

        end_time = time.time()
        embedding_time = end_time - start_time

        embedding_details_path = os.path.join(self.exp_dir, "detailed_results.json")
        with open(embedding_details_path, 'w') as f:
            json.dump({'embedding_time': embedding_time}, f)

    def ir_exp(self):
        self.logger.info("Starting IR experiment...")

        self.data_config = self.config.get('data', {})

        agent_class = search_agent.AGENT_CLASSES[self.config.get('agent', {}).get('agent_class')]
        self.agent = agent_class(self.config)

        self.results_dir = Path(self.exp_dir) / 'per_query_results'
        self.results_dir.mkdir(exist_ok=True)

        queries_path = self.data_config.get('q_text_csv', '')
        qs_df = pd.read_csv(queries_path, header=0)
        qids = qs_df.iloc[:, 0].tolist()
        queries = qs_df.iloc[:, 1].tolist()

        for qid, query in zip(qids, queries):
            try:
                self.logger.info(f"Ranking query {qid}: {query}")

                result = self.agent.act(query,qid)

                if result['top_k_psgs']:
                    self.logger.info('Rank successful')
                    self.write_query_result(qid, result)
                else:
                    self.logger.error(f'Failed to rank query {qid} -- empty result[\'top_k_psgs\']')

            except Exception as e:
                self.logger.error(f'Failed to rank or write results for query {qid}: {str(e)}')
            
    def write_query_result(self, qid, result):
        """
        Write two files: 
        1) TREC run file : trec_results_raw.txt (may have duplicates from LLM reranking)
        2) JSON: detailed_results.json
        """
        query_result_dir = self.results_dir / f"{qid}"
        query_result_dir.mkdir(exist_ok=True)
        detailed_results_path = query_result_dir / "detailed_results.json"
        trec_file_path = query_result_dir / "trec_results_raw.txt"

        with open(detailed_results_path, 'w') as file:
            json.dump(result, file, indent=4)

        trec_results = []
        top_k_psgs = result.get('top_k_psgs', [])
        for p_rank, pid in enumerate(top_k_psgs):
            score = len(top_k_psgs) - p_rank
            trec_results.append(f"{qid} Q0 {pid} {p_rank + 1} {score} run")

        with open(trec_file_path, "w") as trec_file:
            trec_file.write("\n".join(trec_results))


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
    parser.add_argument("-c", "--exp-dir", type=str, required=True, help="Path to the experiment directory containing config.yaml")
    parser.add_argument("-e", "--exp-type", type=str, required=True, help="Name of the experiment method to run (e.g., index_corpus)")
    args = parser.parse_args()

    load_dotenv()

    config_path = os.path.join(args.exp_dir, "config.yaml")
    if not os.path.exists(config_path):
        print(f"No config.yaml found in {args.exp_dir}. Skipping experiment.")
    else:
        manager = ExperimentManager(args.exp_dir)
        manager.run_experiment(args.exp_type)
