import os
import re
import yaml
import argparse
import pytrec_eval
import json
import numpy as np
from scipy.stats import norm
from fpdf import FPDF
from nl_pe.utils.setup_logging import setup_logging
from nl_pe.utils.utils import get_doc_text_list
import unicodedata
from pathlib import Path
from nl_pe.llm.prompter import Prompter
from sklearn.metrics import average_precision_score
from sklearn.linear_model import LinearRegression
import warnings



class EvalManager:
    def __init__(self, eval_dir, skip_existing=False):
        self.eval_dir = eval_dir
        self.skip_existing = skip_existing
        self.load_config()
        self.setup_logger()

        self.selected_trec_measures = self.config.get("measures", pytrec_eval.supported_measures)
        self.qrels_path = self.config.get("qrels_path")
        self.qrels_dict = self.load_pytrec_eval_qrels(self.qrels_path)

        self.results_dir = Path(self.eval_dir) / "per_query_results"
        self.all_query_trec_eval_results = {}
        self.all_eval_results_path = Path(self.eval_dir) / "all_queries_eval_results.jsonl"

    def load_config(self):
        config_path = os.path.join(self.eval_dir, "eval_config.yaml")
        with open(config_path, "r") as config_file:
            self.config = yaml.safe_load(config_file)

    def setup_logger(self):
        self.logger = setup_logging(self.__class__.__name__, self.config, output_file=os.path.join(self.eval_dir, "evaluation.log"))

    def evaluate_experiment(self):
        self.logger.info("Starting evaluation...")

        # Check if all required output files exist
        required_files = [
            self.all_eval_results_path,
        ]

        if self.skip_existing and all(file.exists() for file in required_files):
            print(f"Skipping evaluation for {self.eval_dir} as all required output files already exist.")
            return

        if not self.results_dir.exists():
            print(f"Results directory {self.results_dir} does not exist")
            return None

        for query_dir in self.results_dir.iterdir():
            if query_dir.is_dir(): 
                self.curr_query_dir = query_dir
                self.curr_qid = query_dir.name
                self.curr_trec_file_path = Path(self.curr_query_dir) / "trec_results_raw.txt"
                self.curr_dedup_trec_file_path = Path(self.curr_query_dir) / "trec_results_deduplicated.txt"
                self.curr_query_eval_results_path = Path(self.curr_query_dir) / "eval_results.jsonl"
                self.curr_query_detailed_results_path = Path(self.curr_query_dir) / "detailed_results.json"

                try:
                    self.evaluate_single_query()
                except Exception as e:
                    self.logger.error(f"Error evaluating query {self.curr_qid}: {e}")

        self.curr_query_dir = None
        self.curr_qid = None
        self.curr_trec_file_path = None
        self.curr_dedup_trec_file_path = None
        self.curr_query_eval_results_path = None
        self.curr_query_detailed_results_path = None

        #aggregate results accross all queries 
        self.write_all_queries_eval_results()

        self.logger.info("Evaluation completed.")

    def evaluate_single_query(self):

        # Remove duplicates from TREC file and save deduplicated version
        deduped_lines = self.deduplicate_trec_results()

        # Parse deduplicated TREC results
        results = pytrec_eval.parse_run(deduped_lines)

        # Evaluate using pytrec_eval
        evaluator = pytrec_eval.RelevanceEvaluator(self.qrels_dict, self.selected_trec_measures)
        per_query_eval_results = evaluator.evaluate(results)

        # Write per-query trec evaluation results
        self.write_query_trec_jsonl(self.curr_query_eval_results_path, per_query_eval_results)

        # Store results for calculating means and std_devs
        self.all_query_trec_eval_results[self.curr_qid] = per_query_eval_results[self.curr_qid]

    def load_pytrec_eval_qrels(self,qrels_path):
        with open(qrels_path, "r") as qrels_file:
            return pytrec_eval.parse_qrel(qrels_file)

    def deduplicate_trec_results(self):
        if not self.curr_trec_file_path.exists() or self.curr_trec_file_path.stat().st_size == 0:
            #if TREC run is empty or missing, write a dummy line
            qid = self.curr_trec_file_path.parent.name
            self.logger.warning(f"Query {qid}: TREC results file {self.curr_trec_file_path} is empty or missing. Adding a dummy line.")
            dummy_line = f"{qid} Q0 dummy_doc_id 1 0.0 dummy_run\n"
            with open(self.curr_dedup_trec_file_path, "w") as dedup_file:
                dedup_file.write(dummy_line)
            return [dummy_line]
        
        with open(self.curr_trec_file_path, "r") as trec_file:
            lines = trec_file.readlines()
            seen_docs = set()
            deduped_lines = []
            for line in lines:
                doc_id = line.split()[2]
                if doc_id not in seen_docs:
                    deduped_lines.append(line)
                    seen_docs.add(doc_id)
                else:
                    self.logger.warning(f"Query {self.curr_trec_file_path} has duplicate doc {doc_id}.")
        with open(self.curr_dedup_trec_file_path, "w") as dedup_file:
            dedup_file.writelines(deduped_lines)

        return deduped_lines

    def write_query_trec_jsonl(self, file_path, data):
        with open(file_path, "w") as file:
            for qid, eval_result in data.items():
                json.dump({"qid": qid, **eval_result}, file)
                file.write("\n")

    def write_all_queries_eval_results(self):
        mean_results = {}
        std_dev_results = {}

        for measure in self.selected_trec_measures:
            values = [result.get(measure) for result in self.all_query_trec_eval_results.values() if result.get(measure) is not None]
            if values:
                mean_value = np.mean(values)
                std_dev = np.std(values, ddof=1) if len(values) > 1 else 0
                mean_results[f"mean_{measure}"] = mean_value
                std_dev_results[f"std_dev_{measure}"] = std_dev

        all_eval_results = {**mean_results, **std_dev_results}
        
        with open(self.all_eval_results_path, "w") as file:
            json.dump(all_eval_results, file)
            file.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate an experiment based on a config file.")
    parser.add_argument("-e", "--eval-dir", type=str, help="The path to the evaluation dir containing eval_config.yaml")
    parser.add_argument("--skip-existing", action="store_true", help="Skip evaluation if output files already exist.")
    args = parser.parse_args()

    eval_manager = EvalManager(args.eval_dir, skip_existing=args.skip_existing)
    eval_manager.evaluate_experiment()
