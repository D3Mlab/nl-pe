import yaml
import json
import argparse
from nl_pe import search_agent
from dotenv import load_dotenv
from nl_pe.dataloading import LOADER_CLASSES
import os
from nl_pe.utils.setup_logging import setup_logging
from pathlib import Path
import math
#import time


class ExperimentManager():

    def __init__(self, exp_dir):
        self.exp_dir = exp_dir
        self.load_config()
        #TODO: update configs to direct subclass loggers to experiment log
        self.setup_logger()

        agent_class = search_agent.AGENT_CLASSES[self.config.get('agent', {}).get('agent_class')]
        self.agent = agent_class(self.config)

        self.data_config = self.config.get('data', {})

        loader_class = LOADER_CLASSES[self.data_config.get('dataloader_class')]
        self.dataloader = loader_class(self.config)

        if not self.data_config.get('unsorted_run_file', False):
            # Ensure the run file is sorted by qid and rank: creates a new path if unsorted
            self.ensure_sorted_run_file()

    def run_experiment(self):
        self.logger.info("Starting experiment...")        

        run_path = self.data_config.get('run_path')

        queries = self.dataloader.get_qs_from_run(run_path)

        self.results_dir = Path(self.exp_dir) / 'per_query_results'
        self.results_dir.mkdir(exist_ok=True)

        #get set of already processed qIDs {q1,q3,...}
        existing_results = self.load_existing_results()
        #temporarily force rerun
        #existing_results = set()

        self.rank_queries(queries, existing_results)

    def rank_queries(self, queries, existing_results):
        for query in queries:
            qid = str(query['qid'])
            if qid in existing_results:
                self.logger.info(f'Results already available for query: {qid}')
                continue

            #a way of running a subset of queries
            if self.config.get('qids_to_run', False):
                if qid not in self.config.get('qids_to_run'):
                    self.logger.info(f"Skipping query {qid} as it's not in the specified list.")
                    continue

            try:
                self.logger.info(f'Ranking query: {qid}')
                passages = self.dataloader.get_psgs_from_run(self.data_config.get('run_path'), qid)

                self.logger.info(f"Truncating passage list to at most {self.data_config.get('k_input',math.inf)} passages")
                passages = passages[:int(self.data_config.get('k_input',math.inf))]

                instance = {
                    'query': query,
                    'psg_list': passages
                }

                #self.logger.debug(f'Ranking query {qid} with instance: {instance}')

                #instance: {
                #           'query': {"qid": __, 'text': __}, 
                #           'psg_list' = [{'pid': __, 'text': __},...]
                #}   

                result = self.agent.rank(instance)

                # dict of agent's final state variables
                # result = {'top_k_psgs': [{pid: __, text: __},...],
                #            ...
                #            <any other final state variables>, e.g. time elapsed, etc
                #            ...
                #            'prev_state_history': [<state_0>, <state_1>, ..., <state_t-1>]} ... each state is a dict (does not include final state)
                #  }

                if result['top_k_psgs']:
                    self.logger.info('Rank successful')
                    self.write_query_result(qid, result)
                else:
                    self.logger.error(f'Failed to rank query {qid} -- empty result[\'top_k_psgs\']')
                #self.write_query_result(qid, result)
            except Exception as e:
                self.logger.error(f'Failed to rank query {qid}: {str(e)}')
                #self.write_query_result(qid, {'error': str(e)})

    def load_existing_results(self):
        """
        Loads existing results by checking for both 'detailed_results.json' and 
        'trec_results_raw.txt' files in the results directory.
        """
        existing_results = set()
        for root, dirs, _ in os.walk(self.results_dir):
           for directory in dirs:
               detailed_results_path = Path(root) / directory / "detailed_results.json"
               trec_results_path = Path(root) / directory / "trec_results_raw.txt"
               if detailed_results_path.exists() and trec_results_path.exists():
                    existing_results.add(directory)  # Directory name is assumed to be the qid
        return existing_results

    def ensure_sorted_run_file(self):
        run_path = self.data_config.get('run_path')
        sorted_run_path = self.dataloader.ensure_sorted_run_file(run_path)
        if run_path != sorted_run_path:
            self.data_config['run_path'] = sorted_run_path
            with open(os.path.join(self.exp_dir, "config.yaml"), "w") as config_file:
                yaml.safe_dump(self.config, config_file)
            self.logger.warning(f"Run file was not sorted. Updated run_path in config to {sorted_run_path}")

    def load_config(self):
        config_path = os.path.join(self.exp_dir, "config.yaml")
        with open(config_path, "r") as config_file:
            self.config = yaml.safe_load(config_file)

    def setup_logger(self):
        self.logger = setup_logging(self.__class__.__name__, self.config, output_file=os.path.join(self.exp_dir, "experiment.log"))

    def write_query_result(self, qid, result):
        """
        Write two files: 
        1) TREC run file : trec_results_raw.txt (may have duplicates from LLM reranking)
        2) JSON: detailed_results.json, with full history of agent states for detailed results (e.g. time elapsed between each state, iterative mprovement in ranking, etc)
        """
        query_result_dir = self.results_dir / f"{qid}"
        query_result_dir.mkdir(exist_ok=True)
        detailed_results_path = query_result_dir / "detailed_results.json"
        trec_file_path = query_result_dir / "trec_results_raw.txt"

        with open(detailed_results_path, 'w') as file:
            json.dump(result, file, indent=4)

        trec_results = []
        top_k_psgs = result.get('top_k_psgs', [])
        for p_index, psg in enumerate(top_k_psgs):
            pid = psg['pid']
            score = len(top_k_psgs) - p_index
            trec_results.append(f"{qid} Q0 {pid} {p_index + 1} {score} llm_reranker_tests")

        with open(trec_file_path, "w") as trec_file:
            trec_file.write("\n".join(trec_results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments in the specified directory.")
    parser.add_argument("-e", "--exp-dir", type=str, help="The path to the experiment dir containing config.yaml")
    args = parser.parse_args()

    load_dotenv()

    config_path = os.path.join(args.exp_dir, "config.yaml")
    if not os.path.exists(config_path):
        print(f"No config.yaml found in {args.exp_dir}. Skipping experiment.")
    else:
        manager = ExperimentManager(args.exp_dir)
        manager.run_experiment()
