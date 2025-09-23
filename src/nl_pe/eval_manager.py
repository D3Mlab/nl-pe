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

        self.binary_qrels_path = self.qrels_path.replace("qrels.txt", "qrels_binary.txt")
        self.binary_qrels_dict = self.load_pytrec_eval_qrels(self.binary_qrels_path)

        self.results_dir = Path(self.eval_dir) / "per_query_results"
        self.all_query_trec_eval_results = {}
        self.all_eval_results_path = Path(self.eval_dir) / "all_queries_eval_results.jsonl"

        self.run_pw_anal = self.config.get("run_pw_anal", False)

        #temporarily set gt relevance threshold to 1 (used in pw analysis only) so 1 or higher is relevant -- incorrect in MSE...
        self.qrels_rel_threshold = 1

        if self.run_pw_anal:
            #initial position confusion
            self.all_query_initial_pw_pos_confusion = {}
            self.max_rel_score = int(self.config.get("min_pw_rel_score"))
            self.initial_pw_pos_confusion_paths = {i: Path(self.eval_dir) / f"all_query_initial_pw_pos_confusion_min_rel_{i}.tsv" for i in range(1, self.max_rel_score + 1)}

            #scores in batch order
            self.all_query_micro_batch_order_scores = {}
            self.micro_batch_order_score_path = Path(self.eval_dir) / "all_query_micro_batch_order_scores.tsv"
            self.macro_batch_order_score_path = Path(self.eval_dir) / "all_query_macro_batch_order_scores.tsv"

            #scores in init order
            self.all_query_micro_init_order_scores = {}
            self.micro_init_order_score_path = Path(self.eval_dir) / "all_query_micro_init_order_scores.tsv"
            self.macro_init_order_score_path = Path(self.eval_dir) / "all_query_macro_init_order_scores.tsv"

        self.all_prompt_runtimes = {}
        self.all_prompt_runtimes_path = Path(self.eval_dir) / "all_total_prompt_runtimes.jsonl"

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
            self.all_prompt_runtimes_path,
        ]
        if self.run_pw_anal:
            required_files.extend([
                self.micro_batch_order_score_path,
                self.macro_batch_order_score_path,
                self.micro_init_order_score_path,
                self.macro_init_order_score_path,
            ])

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

        self.all_prompt_runtimes[self.curr_qid] = self.get_query_prompt_runtime()

        if self.run_pw_anal:
            if not os.path.exists(self.curr_query_detailed_results_path):
                self.logger.debug(f"Query {self.curr_qid}: detailed_results.json file is missing.")
            else:

                #initial position confusion
                try:
                    #e.g. self.all_query_initial_pos_confusion = {'q1': {3: {'ranks': [1,2,3...], 'gt' : [1,0,1...], 'tp': [1, 0, 0, ...],
                    #                                                   'fp': [0, 1, 0, ...], 'fn': [0, 0, 1, ...], tn: [0, 0, 0, ...]}
                    #                                                    2: {...}},
                    #                                             'q2': {...}, ...}
                    self.all_query_initial_pw_pos_confusion[self.curr_query_dir.name] = self.get_per_query_pw_initial_pos_confusion()
                except Exception as e:
                    self.logger.error(f"Error calculating initial pos confusion for query {self.curr_qid}: {e}")

                #batch order scores
                try:
                    #e.g. {'q1': {0: {all_scores: [<pos 0 score 0>, <pos 0 score 1>, ...],
                    #              gt_pos_scores: [<pos 0 gt pos score 0>,...],
                    #              gt_neg_scores: [<pos 0 gt neg score 0>,...]},
                    #              gt_pid_rels: [<pos 0 gt pid rel 0>, ...]},
                    #             1: {<pos 1 dict>},
                    #             ...
                    #             },
                    #        'q2': {0:{<pos 0 dict>}, 1:{<pos 1 dict>}, ...}
                    #       }
                    # note there is no important order to the scores given a position,
                    # it is just the LLM call order and there could be no scores at all (e.g. hallucinations, no positive scores)
                    #
                    self.all_query_micro_batch_order_scores[self.curr_query_dir.name] = self.get_per_query_batch_order_scores()
                except Exception as e:
                    self.logger.error(f"Error collecting batch scores for query {self.curr_qid}: {e}")

                #initial order scores
                try:
                    #same data structure as batch order scores above
                    self.all_query_micro_init_order_scores[self.curr_query_dir.name] = self.get_per_query_initial_position_scores()
                except Exception as e:
                    self.logger.error(f"Error collecting init order scores for query {self.curr_qid}: {e}")

    def get_query_prompt_runtime(self):
        
        if self.curr_query_detailed_results_path.exists():
            with open(self.curr_query_detailed_results_path, "r", encoding="utf-8") as f:
                detailed_results = json.load(f)
        else:
            return None
        
        return np.array(detailed_results.get("prompting_runtimes", [])).sum()

    def get_per_query_batch_order_scores(self):
        with open(self.curr_query_detailed_results_path, "r", encoding="utf-8") as f:
            detailed_results = json.load(f)

        #e.g. [['p1','p2','p3'], ['p3','p2','p1']]
        batch_pid_history = detailed_results.get("batch_pid_history", [])
        #e.g. [[3,2,3], [2,3,1]]
        batch_scores = detailed_results.get("batch_scores", [])
        
        batch_size = len(batch_pid_history[0])

        prompt_token_history = detailed_results.get("prompt_tokens", [0])

        micro_query_results = {
            'mean_prompt_tokens': np.mean(prompt_token_history)
        }

        for i, batch in enumerate(batch_scores):
            for j, score in enumerate(batch):

                #don't consider any hallucinated scores beyond input list length
                if j > batch_size-1:
                    continue
                
                if j not in micro_query_results:
                    micro_query_results[j] = {
                        'all_scores': [],
                        'gt_pos_scores': [],
                        'gt_neg_scores': [],
                        'gt_pid_rels': [],
                        'gt_binary_pid_rels': []#,
                        #'prompt_tokens': []
                    }

                #cap rare hallucinated scores (nova-pro only) at 3
                if score > 3:
                    score = 3

                micro_query_results[j]['all_scores'].append(score)

                try:
                    pid = batch_pid_history[i][j]
                    #prompt_tokens = prompt_token_history[i]
                    #micro_query_results[j]['prompt_tokens'].append(prompt_tokens)
                except IndexError:
                    #self.logger.debug(f"Missing PID for query {self.curr_query_dir.name} batch {i}, position {j}. Skipping.")
                    #these errors will come up if the LLM produces more scores than passages
                    continue

                pid_q_rel = self.qrels_dict.get(self.curr_qid, {}).get(pid, 0)
                micro_query_results[j]['gt_pid_rels'].append(pid_q_rel)

                pid_binary_q_rel = self.binary_qrels_dict.get(self.curr_qid, {}).get(pid, 0)
                micro_query_results[j]['gt_binary_pid_rels'].append(pid_binary_q_rel)

                if pid_binary_q_rel == 1:
                    micro_query_results[j]['gt_pos_scores'].append(score)
                elif pid_binary_q_rel == 0:
                    micro_query_results[j]['gt_neg_scores'].append(score)
                else:
                    raise ValueError(f"Unexpected binary relevance value {pid_binary_q_rel} for PID {pid} in query {self.curr_qid}")

                # Get the prompt tokens for the current batch

        return micro_query_results

    def get_per_query_initial_position_scores(self):
        with open(self.curr_query_detailed_results_path, "r", encoding="utf-8") as f:
            detailed_results = json.load(f)

        pid_list = self.get_init_pid_list(detailed_results)
        pid_to_score_dict = detailed_results['pid_to_score_dict']

        prompt_token_history = detailed_results.get("prompt_tokens", [0])

        micro_query_results = {
            'mean_prompt_tokens': np.mean(prompt_token_history)
        }

        for i, pid in enumerate(pid_list):
            if i not in micro_query_results:
                micro_query_results[i] = {
                    'all_scores': [],
                    'gt_pos_scores': [],
                    'gt_neg_scores': [],
                    'gt_pid_rels': [],
                    'gt_binary_pid_rels': []#,
                    #'prompt_tokens': []
                }

            if pid_to_score_dict.get(pid, None):
                pid_scores = pid_to_score_dict[pid]
                #cap rare hallucinated scores (nova-pro only) at 3
                pid_scores = [pid_score if pid_score <= 3 else 3 for pid_score in pid_scores]
            else:
                pid_scores = [0]

            micro_query_results[i]['all_scores'].extend(pid_scores)

            pid_q_rels = [self.qrels_dict.get(self.curr_qid, {}).get(pid, 0)] * len(pid_scores)
            micro_query_results[i]['gt_pid_rels'].extend(pid_q_rels)

            pid_binary_q_rels = [self.binary_qrels_dict.get(self.curr_qid, {}).get(pid, 0)] * len(pid_scores)
            micro_query_results[i]['gt_binary_pid_rels'].extend(pid_binary_q_rels)

            if pid_binary_q_rels[0] == 1:
                micro_query_results[i]['gt_pos_scores'].extend(pid_scores)
            elif pid_binary_q_rels[0] == 0:
                micro_query_results[i]['gt_neg_scores'].extend(pid_scores)
            else:
                raise ValueError(f"Unexpected binary relevance value {pid_binary_q_rels[0]} for PID {pid} in query {self.curr_qid}")


        return micro_query_results       


    def get_per_query_pw_initial_pos_confusion(self): 
        with open(self.curr_query_detailed_results_path, "r", encoding="utf-8") as f:
            detailed_results = json.load(f)

        pid_list = self.get_init_pid_list(detailed_results)
        scores = self.get_init_pos_scores(detailed_results)
        gt = [1 if self.qrels_dict.get(self.curr_qid, {}).get(pid, 0) >= self.qrels_rel_threshold else 0 for pid in pid_list]

        confusion_data = {}
        for rel_score in range(1, self.max_rel_score + 1):
            tp = [1 if g == 1 and s is not None and s >= rel_score else 0 for g, s in zip(gt, scores)]
            fp = [1 if g == 0 and s is not None and s >= rel_score else 0 for g, s in zip(gt, scores)]
            fn = [1 if g == 1 and s is not None and s < rel_score else 0 for g, s in zip(gt, scores)]
            tn = [1 if g == 0 and s is not None and s < rel_score else 0 for g, s in zip(gt, scores)]
            hall = [1 if s is None else 0 for s in scores]


            confusion_data[rel_score] = {
                'ranks': list(range(1, min(len(scores), len(pid_list)) + 1)),
                'gt': gt,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'tn': tn,
                'hall': hall,
                'P': [tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) > 0 else 0 for i in range(len(tp))],
                'R': [tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) > 0 else 0 for i in range(len(tp))]
            }

            # Save to pw_initial_pos_confusion.tsv
            tsv_path = self.curr_query_dir / f"pw_initial_pos_confusion_min_rel_{rel_score}.tsv"
            with open(tsv_path, "w") as tsv_file:
                tsv_file.write("rank\tgt\ttp\tfp\tfn\ttn\thall\tP\tR\n")
                for i in range(min(len(scores), len(pid_list))):
                    tsv_file.write(f"{confusion_data[rel_score]['ranks'][i]}\t{confusion_data[rel_score]['gt'][i]}\t{confusion_data[rel_score]['tp'][i]}\t{confusion_data[rel_score]['fp'][i]}\t{confusion_data[rel_score]['fn'][i]}\t{confusion_data[rel_score]['tn'][i]}\t{confusion_data[rel_score]['P'][i]:.3f}\t{confusion_data[rel_score]['hall'][i]}\t{confusion_data[rel_score]['R'][i]:.3f}\n")

        return confusion_data

    def get_init_pid_list(self, detailed_results):
        return [d['pid'] for d in detailed_results["instance"]['psg_list']]
    
    def get_init_pos_scores(self, detailed_results):
        if detailed_results.get('scores_init_order'):
            return detailed_results['scores_init_order']
        elif detailed_results.get('responses'):
            scores = []
            for response in detailed_results['responses']:
                try: 
                    scores.extend(int(score) for score in self.parse_llm_list(response))
                except Exception as e:
                    self.logger.error(f"Error parsing scores from response: {e}")
                    return None
            return scores
        else:
            return None

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

        if self.all_prompt_runtimes:
            with open(self.all_prompt_runtimes_path, "w") as file:
                 json.dump(self.all_prompt_runtimes, file)
                 file.write("\n")    

        #pw related eval
        if self.run_pw_anal:

            #initial position confusion
            try:
                self.write_all_query_pw_initial_pos_confusion()
            except Exception as e:
                self.logger.error(f"Error writing initial pos confusion for all queries: {e}")

            #order scores
            try:    
                self.write_all_query_order_scores()
            except Exception as e:
                self.logger.error(f"Error writing order scores for all queries: {e}")

    def write_all_query_order_scores(self):
        if not self.all_query_micro_batch_order_scores and not self.all_query_micro_init_order_scores:
            return None

        def process_order_scores(order_scores):
            micro_order_score_data = {}
            macro_order_score_data = {
                'mean_prompt_tokens': [],
            }

            for query_scores in order_scores.values():
                macro_order_score_data['mean_prompt_tokens'].append(query_scores.pop('mean_prompt_tokens', None))
                for pos, pos_data in query_scores.items():
                    if pos not in micro_order_score_data:
                        micro_order_score_data[pos] = {
                            'all_scores': [],
                            'gt_pos_scores': [],
                            'gt_neg_scores': [],
                            'gt_pid_rels': [],
                            'gt_binary_pid_rels': []
                        }

                    if pos not in macro_order_score_data:
                        macro_order_score_data[pos] = {
                            'avg_scores': [],
                            'scores_var': [],
                            'scores_std': [],
                            'scores_q3': [],
                            'scores_q1': [],
                            'scores_median': [],
                            'avg_gt_pos_scores': [],
                            'avg_binary_pid_rels': [],
                            'binary_pid_rels_var': [],
                            'binary_pid_rels_std': [],
                            'binary_pid_rels_q3': [],
                            'binary_pid_rels_q1': [],
                            'binary_pid_rels_median': [],
                            'gt_pos_scores_var': [],
                            'gt_pos_scores_std': [],
                            'gt_pos_scores_q3': [],
                            'gt_pos_scores_q1': [],
                            'gt_pos_scores_median': [],
                            'avg_gt_neg_scores': [],
                            'gt_neg_scores_var': [],
                            'gt_neg_scores_std': [],
                            'gt_neg_scores_q3': [],
                            'gt_neg_scores_q1': [],
                            'gt_neg_scores_median': [],
                        }

                    # Micro
                    micro_order_score_data[pos]['all_scores'].extend(pos_data.get('all_scores', []))
                    micro_order_score_data[pos]['gt_pos_scores'].extend(pos_data.get('gt_pos_scores', []))
                    micro_order_score_data[pos]['gt_neg_scores'].extend(pos_data.get('gt_neg_scores', []))
                    micro_order_score_data[pos]['gt_pid_rels'].extend(pos_data.get('gt_pid_rels', []))
                    micro_order_score_data[pos]['gt_binary_pid_rels'].extend(pos_data.get('gt_binary_pid_rels', []))

                    # Macro
                    macro_order_score_data[pos]['avg_scores'].append(np.mean(pos_data.get('all_scores', []))) if pos_data.get('all_scores', []) else 0
                    macro_order_score_data[pos]['scores_var'].append(np.var(pos_data.get('all_scores', []), ddof=1)) if len(pos_data.get('all_scores', [])) > 1 else 0
                    macro_order_score_data[pos]['scores_std'].append(np.std(pos_data.get('all_scores', []), ddof=1)) if len(pos_data.get('all_scores', [])) > 1 else 0
                    if len(pos_data.get('all_scores', [])) > 1:
                        macro_order_score_data[pos]['scores_q3'].append(np.percentile(pos_data.get('all_scores', []), 75))
                        macro_order_score_data[pos]['scores_q1'].append(np.percentile(pos_data.get('all_scores', []), 25))
                        macro_order_score_data[pos]['scores_median'].append(np.median(pos_data.get('all_scores', [])))
                    elif len(pos_data.get('all_scores', [])) == 1:
                        macro_order_score_data[pos]['scores_q3'].append(pos_data['all_scores'][0])
                        macro_order_score_data[pos]['scores_q1'].append(pos_data['all_scores'][0])
                        macro_order_score_data[pos]['scores_median'].append(pos_data['all_scores'][0])

                    macro_order_score_data[pos]['avg_gt_pos_scores'].append(np.mean(pos_data.get('gt_pos_scores', []))) if pos_data.get('gt_pos_scores', []) else 0
                    macro_order_score_data[pos]['gt_pos_scores_var'].append(np.var(pos_data.get('gt_pos_scores', []), ddof=1)) if len(pos_data.get('gt_pos_scores', [])) > 1 else 0
                    macro_order_score_data[pos]['gt_pos_scores_std'].append(np.std(pos_data.get('gt_pos_scores', []), ddof=1)) if len(pos_data.get('gt_pos_scores', [])) > 1 else 0
                    if len(pos_data.get('gt_pos_scores', [])) > 1:
                        macro_order_score_data[pos]['gt_pos_scores_q3'].append(np.percentile(pos_data.get('gt_pos_scores', []), 75))
                        macro_order_score_data[pos]['gt_pos_scores_q1'].append(np.percentile(pos_data.get('gt_pos_scores', []), 25))
                        macro_order_score_data[pos]['gt_pos_scores_median'].append(np.median(pos_data.get('gt_pos_scores', [])))
                    elif len(pos_data.get('gt_pos_scores', [])) == 1:
                        macro_order_score_data[pos]['gt_pos_scores_q3'].append(pos_data['gt_pos_scores'][0])
                        macro_order_score_data[pos]['gt_pos_scores_q1'].append(pos_data['gt_pos_scores'][0])
                        macro_order_score_data[pos]['gt_pos_scores_median'].append(pos_data['gt_pos_scores'][0])

                    macro_order_score_data[pos]['avg_gt_neg_scores'].append(np.mean(pos_data.get('gt_neg_scores', []))) if pos_data.get('gt_neg_scores', []) else 0
                    macro_order_score_data[pos]['gt_neg_scores_var'].append(np.var(pos_data.get('gt_neg_scores', []), ddof=1)) if len(pos_data.get('gt_neg_scores', [])) > 1 else 0
                    macro_order_score_data[pos]['gt_neg_scores_std'].append(np.std(pos_data.get('gt_neg_scores', []), ddof=1)) if len(pos_data.get('gt_neg_scores', [])) > 1 else 0
                    if len(pos_data.get('gt_neg_scores', [])) > 1:
                        macro_order_score_data[pos]['gt_neg_scores_q3'].append(np.percentile(pos_data.get('gt_neg_scores', []), 75))
                        macro_order_score_data[pos]['gt_neg_scores_q1'].append(np.percentile(pos_data.get('gt_neg_scores', []), 25))
                        macro_order_score_data[pos]['gt_neg_scores_median'].append(np.median(pos_data.get('gt_neg_scores', [])))
                    elif len(pos_data.get('gt_neg_scores', [])) == 1:
                        macro_order_score_data[pos]['gt_neg_scores_q3'].append(pos_data['gt_neg_scores'][0])
                        macro_order_score_data[pos]['gt_neg_scores_q1'].append(pos_data['gt_neg_scores'][0])
                        macro_order_score_data[pos]['gt_neg_scores_median'].append(pos_data['gt_neg_scores'][0])

                    macro_order_score_data[pos]['avg_binary_pid_rels'].append(np.mean(pos_data.get('gt_binary_pid_rels', []))) if pos_data.get('gt_binary_pid_rels', []) else 0
                    macro_order_score_data[pos]['binary_pid_rels_var'].append(np.var(pos_data.get('gt_binary_pid_rels', []), ddof=1)) if len(pos_data.get('gt_binary_pid_rels', [])) > 1 else 0
                    macro_order_score_data[pos]['binary_pid_rels_std'].append(np.std(pos_data.get('gt_binary_pid_rels', []), ddof=1)) if len(pos_data.get('gt_binary_pid_rels', [])) > 1 else 0
                    if len(pos_data.get('gt_binary_pid_rels', [])) > 1:
                        macro_order_score_data[pos]['binary_pid_rels_q3'].append(np.percentile(pos_data.get('gt_binary_pid_rels', []), 75))
                        macro_order_score_data[pos]['binary_pid_rels_q1'].append(np.percentile(pos_data.get('gt_binary_pid_rels', []), 25))
                        macro_order_score_data[pos]['binary_pid_rels_median'].append(np.median(pos_data.get('gt_binary_pid_rels', [])))
                    elif len(pos_data.get('gt_binary_pid_rels', [])) == 1:
                        macro_order_score_data[pos]['binary_pid_rels_q3'].append(pos_data['gt_binary_pid_rels'][0])
                        macro_order_score_data[pos]['binary_pid_rels_q1'].append(pos_data['gt_binary_pid_rels'][0])
                        macro_order_score_data[pos]['binary_pid_rels_median'].append(pos_data['gt_binary_pid_rels'][0])

            macro_order_score_data['mean_prompt_tokens'] = np.mean(macro_order_score_data['mean_prompt_tokens']) if macro_order_score_data['mean_prompt_tokens'] else 0

            return micro_order_score_data, macro_order_score_data

        def write_tsv(file_path, header, data, is_micro):
            with open(file_path, "w") as tsv_file:
                tsv_file.write(header)
                
                mean_all_pos_prompt_tokens = round(data.pop('mean_prompt_tokens',0))

                for pos, pos_data in sorted(data.items()):
                    if is_micro:
                        all_scores = pos_data['all_scores']
                        n_scores = len(all_scores)
                        mean_score = np.mean(all_scores) if all_scores else 0
                        var_score = np.var(all_scores, ddof=1) if len(all_scores) > 1 else 0
                        mse_all = np.mean((np.array(pos_data['all_scores']) - np.array(pos_data['gt_pid_rels']))**2) if pos_data['all_scores'] and pos_data['gt_pid_rels'] else 0
                        auc_pr_all = self.auc_pr_from_micro_scores(all_scores, pos_data['gt_binary_pid_rels'])
                        micro_f1_all = self.micro_f1_from_micro_scores(all_scores, pos_data['gt_binary_pid_rels'])

                        gt_pos_scores = pos_data['gt_pos_scores']
                        n_gt_pos_scores = len(gt_pos_scores)
                        mean_gt_pos_score = np.mean(gt_pos_scores) if gt_pos_scores else 0
                        var_gt_pos_score = np.var(gt_pos_scores, ddof=1) if len(gt_pos_scores) > 1 else 0

                        gt_neg_scores = pos_data['gt_neg_scores']
                        n_gt_neg_scores = len(gt_neg_scores)
                        mean_gt_neg_score = np.mean(gt_neg_scores) if gt_neg_scores else 0
                        var_gt_neg_score = np.var(gt_neg_scores, ddof=1) if len(gt_neg_scores) > 1 else 0

                        se_pos = []
                        se_neg = []
                        for gt_label, score in zip(pos_data['gt_pid_rels'], pos_data['all_scores']):
                            if gt_label >= self.qrels_rel_threshold:
                                se_pos.append((gt_label - score)**2)
                            else:
                                se_neg.append((gt_label - score)**2)
                        mse_pos = np.mean(se_pos) if se_pos else 0
                        mse_neg = np.mean(se_neg) if se_neg else 0

                        # Append slope and intercept to each row
                        tsv_file.write(f"{pos}\
                                       \t{n_scores:d}\
                                       \t{mean_score:.8f}\
                                       \t{var_score:.8f}\
                                       \t{n_gt_pos_scores:d}\
                                       \t{mean_gt_pos_score:.8f}\
                                       \t{var_gt_pos_score:.8f}\
                                       \t{n_gt_neg_scores:d}\
                                       \t{mean_gt_neg_score:.3f}\
                                       \t{var_gt_neg_score:.3f}\
                                       \t{mse_all:.8f}\
                                       \t{mse_pos:.8f}\
                                       \t{mse_neg:.8f}\
                                       \t{auc_pr_all:.8f}\
                                       \t{micro_f1_all:.8f}\n")
                    else:
                        mean_avg_scores = np.mean(pos_data['avg_scores']) if pos_data['avg_scores'] else 0
                        mean_scores_var = np.mean(pos_data['scores_var']) if pos_data['scores_var'] else 0
                        mean_scores_std = np.mean(pos_data['scores_std']) if pos_data['scores_std'] else 0
                        std_scores_std = np.std(pos_data['scores_std'], ddof=1) if len(pos_data['scores_std']) > 1 else 0
                        mean_scores_q3 = np.mean(pos_data['scores_q3']) if pos_data['scores_q3'] else 0
                        mean_scores_q1 = np.mean(pos_data['scores_q1']) if pos_data['scores_q1'] else 0
                        mean_scores_median = np.mean(pos_data['scores_median']) if pos_data['scores_median'] else 0

                        mean_avg_gt_pos_scores = np.mean(pos_data['avg_gt_pos_scores']) if pos_data['avg_gt_pos_scores'] else 0
                        mean_gt_pos_scores_var = np.mean(pos_data['gt_pos_scores_var']) if pos_data['gt_pos_scores_var'] else 0
                        mean_gt_pos_scores_std = np.mean(pos_data['gt_pos_scores_std']) if pos_data['gt_pos_scores_std'] else 0
                        std_gt_pos_scores_std = np.std(pos_data['gt_pos_scores_std'], ddof=1) if len(pos_data['gt_pos_scores_std']) > 1 else 0
                        mean_gt_pos_scores_q3 = np.mean(pos_data['gt_pos_scores_q3']) if pos_data['gt_pos_scores_q3'] else 0
                        mean_gt_pos_scores_q1 = np.mean(pos_data['gt_pos_scores_q1']) if pos_data['gt_pos_scores_q1'] else 0
                        mean_gt_pos_scores_median = np.mean(pos_data['gt_pos_scores_median']) if pos_data['gt_pos_scores_median'] else 0

                        mean_avg_gt_neg_scores = np.mean(pos_data['avg_gt_neg_scores']) if pos_data['avg_gt_neg_scores'] else 0
                        mean_gt_neg_scores_var = np.mean(pos_data['gt_neg_scores_var']) if pos_data['gt_neg_scores_var'] else 0
                        mean_gt_neg_scores_std = np.mean(pos_data['gt_neg_scores_std']) if pos_data['gt_neg_scores_std'] else 0
                        std_gt_neg_scores_std = np.std(pos_data['gt_neg_scores_std'], ddof=1) if len(pos_data['gt_neg_scores_std']) > 1 else 0
                        mean_gt_neg_scores_q3 = np.mean(pos_data['gt_neg_scores_q3']) if pos_data['gt_neg_scores_q3'] else 0
                        mean_gt_neg_scores_q1 = np.mean(pos_data['gt_neg_scores_q1']) if pos_data['gt_neg_scores_q1'] else 0
                        mean_gt_neg_scores_median  = np.mean(pos_data['gt_neg_scores_median']) if pos_data['gt_neg_scores_median'] else 0

                        mean_avg_binary_pid_rels = np.mean(pos_data['avg_binary_pid_rels']) if pos_data['avg_binary_pid_rels'] else 0
                        mean_binary_pid_rels_var = np.mean(pos_data['binary_pid_rels_var']) if pos_data['binary_pid_rels_var'] else 0
                        mean_binary_pid_rels_std = np.mean(pos_data['binary_pid_rels_std']) if pos_data['binary_pid_rels_std'] else 0
                        mean_binary_pid_rels_q3 = np.mean(pos_data['binary_pid_rels_q3']) if pos_data['binary_pid_rels_q3'] else 0
                        mean_binary_pid_rels_q1 = np.mean(pos_data['binary_pid_rels_q1']) if pos_data['binary_pid_rels_q1'] else 0
                        mean_binary_pid_rels_median = np.mean(pos_data['binary_pid_rels_median']) if pos_data['binary_pid_rels_median'] else 0

                        # Calculate macro F1 score
                        macro_f1_score = self.macro_f1_for_pos(pos_data)
                        macro_auc_score = self.macro_auc_for_pos(pos_data)
                        macro_mse_all = self.macro_mse_for_pos(pos_data)

                        tsv_file.write(f"{pos}\t{mean_avg_scores:.8f}\
                                       \t{mean_scores_var:.8f}\
                                       \t{mean_scores_std:.8f}\
                                       \t{mean_scores_q3:.8f}\
                                       \t{mean_scores_q1:.8f}\
                                       \t{mean_scores_median:.8f}\
                                       \t{mean_avg_gt_pos_scores:.8f}\
                                       \t{mean_gt_pos_scores_var:.8f}\
                                       \t{mean_gt_pos_scores_std:.8f}\
                                       \t{mean_gt_pos_scores_q3:.8f}\
                                       \t{mean_gt_pos_scores_q1:.8f}\
                                       \t{mean_gt_pos_scores_median:.8f}\
                                       \t{mean_avg_gt_neg_scores:.8f}\
                                       \t{mean_gt_neg_scores_var:.8f}\
                                       \t{mean_gt_neg_scores_std:.8f}\
                                       \t{mean_gt_neg_scores_q3:.8f}\
                                       \t{mean_gt_neg_scores_q1:.8f}\
                                       \t{mean_gt_neg_scores_median:.8f}\
                                       \t{mean_avg_binary_pid_rels:.8f}\
                                       \t{mean_binary_pid_rels_var:.8f}\
                                       \t{mean_binary_pid_rels_std:.8f}\
                                       \t{mean_binary_pid_rels_q3:.8f}\
                                       \t{mean_binary_pid_rels_q1:.8f}\
                                       \t{mean_binary_pid_rels_median:.8f}\
                                       \t{std_scores_std:.8f}\
                                       \t{std_gt_pos_scores_std:.8f}\
                                       \t{std_gt_neg_scores_std:.8f}\
                                       \t{mean_all_pos_prompt_tokens:d}\
                                       \t{macro_f1_score:.8f}\
                                       \t{macro_auc_score:.8f}\
                                       \t{macro_mse_all:.8f}\
                                       \n") 

                        
        # Define headers as strings before the function call
        micro_header = "pos\tn_scores\tmean_score\tvar_score\
\tn_gt_pos_scores\tmean_gt_pos_score\tvar_gt_pos_score\
\tn_gt_neg_scores\tmean_gt_neg_score\tvar_gt_neg_scor\
\tmse_all\tmse_pos\tmse_neg\
\tauc_pr_all\tmicro_f1_all\n"
        
        macro_header = "pos\tmean_avg_scores\tmean_scores_var\tmean_scores_std\tmean_scores_q3\tmean_scores_q1\tmean_scores_median\
\tmean_avg_gt_pos_scores\tmean_gt_pos_scores_var\tmean_gt_pos_scores_std\tmean_gt_pos_scores_q3\tmean_gt_pos_scores_q1\tmean_gt_pos_scores_median\
\tmean_avg_gt_neg_scores\tmean_gt_neg_scores_var\tmean_gt_neg_scores_std\tmean_gt_neg_scores_q3\tmean_gt_neg_scores_q1\tmean_gt_neg_scores_median\
\tmean_avg_binary_pid_rels\tmean_binary_pid_rels_var\tmean_binary_pid_rels_std\tmean_binary_pid_rels_q3\tmean_binary_pid_rels_q1\tmean_binary_pid_rels_median\
\tstd_scores_std\tstd_gt_pos_scores_std\tstd_gt_neg_scores_std\
\tmean_all_pos_prompt_tokens\
\tmacro_f1_score\
\tmacro_auc_score\
\tmacro_mse_all\
\n" 

        # Process and write batch order scores
        micro_batch_data, macro_batch_data = process_order_scores(self.all_query_micro_batch_order_scores)
        write_tsv(self.micro_batch_order_score_path, micro_header, micro_batch_data, True)
        write_tsv(self.macro_batch_order_score_path, macro_header, macro_batch_data, False)

        # Process and write init order scores
        micro_init_data, macro_init_data = process_order_scores(self.all_query_micro_init_order_scores)
        write_tsv(self.micro_init_order_score_path, micro_header, micro_init_data, True)
        write_tsv(self.macro_init_order_score_path, macro_header, macro_init_data, False)

    def write_all_query_pw_initial_pos_confusion(self):
        if not self.all_query_initial_pw_pos_confusion:
            return None
        
        
        rank_data = {i: {} for i in range(1, self.max_rel_score + 1)}
        
        for query_confusion in self.all_query_initial_pw_pos_confusion.values():
            for rel_score in range(1, self.max_rel_score + 1):
                for rank, gt, tp, fp, fn, tn, hall in zip(query_confusion[rel_score]['ranks'], query_confusion[rel_score]['gt'], query_confusion[rel_score]['tp'], query_confusion[rel_score]['fp'], query_confusion[rel_score]['fn'], query_confusion[rel_score]['tn'], query_confusion[rel_score]['hall']):
                    if rank not in rank_data[rel_score]:
                        rank_data[rel_score][rank] = {'n_queries': 0, 'gt': 0, 'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0, 'hall': 0}
                    rank_data[rel_score][rank]['n_queries'] += 1
                    rank_data[rel_score][rank]['gt'] += gt
                    rank_data[rel_score][rank]['tp'] += tp
                    rank_data[rel_score][rank]['fp'] += fp
                    rank_data[rel_score][rank]['fn'] += fn
                    rank_data[rel_score][rank]['tn'] += tn
                    rank_data[rel_score][rank]['hall'] += hall 


        for rel_score in range(1, self.max_rel_score + 1):
            with open(self.initial_pw_pos_confusion_paths[rel_score], "w") as tsv_file:
                tsv_file.write("rank\tn_qs\tfrac_tp+fn\tfrac_tp\tfrac_fp\tfrac_fn\tfrac_tn\tfrac_hall\tP\tR\tFPR\tF1\n")
                n_queries_set = set()
                for rank in sorted(rank_data[rel_score].keys()):
                    n_queries = rank_data[rel_score][rank]['n_queries']
                    # track unique values of n_queries to see if consistent across all ranks
                    n_queries_set.add(n_queries)
                    frac_gt = rank_data[rel_score][rank]['gt'] / n_queries
                    frac_tp = rank_data[rel_score][rank]['tp'] / n_queries
                    frac_fp = rank_data[rel_score][rank]['fp'] / n_queries
                    frac_fn = rank_data[rel_score][rank]['fn'] / n_queries
                    frac_tn = rank_data[rel_score][rank]['tn'] / n_queries
                    frac_hall = rank_data[rel_score][rank]['hall'] / n_queries
                    P = frac_tp / (frac_tp + frac_fp) if (frac_tp + frac_fp) > 0 else 0
                    R = frac_tp / (frac_tp + frac_fn) if (frac_tp + frac_fn) > 0 else 0
                    FPR = frac_fp / (frac_tn + frac_fp) if (frac_tn + frac_fp) > 0 else 0
                    F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0
                    tsv_file.write(f"{rank}\t{n_queries}\t{frac_gt:.3f}\t{frac_tp:.3f}\t{frac_fp:.3f}\t{frac_fn:.3f}\t{frac_tn:.3f}\t{frac_hall:.3f}\t{P:.3f}\t{R:.3f}\t{FPR:.3f}\t{F1:.3f}\n")

                if len(n_queries_set) > 1:
                    self.logger.warning(f"Not all n_queries values are the same across all rows for rel_score {rel_score}.")

    def auc_pr_from_micro_scores(self, scores, labels):

        #convert 0-3 scores to probabilities
        probs = np.clip(np.array(scores), 0, 3) / 3.0
        labels = np.array(labels)

        #hallucinations!
        if len(probs) != len(labels):
            print(f"Length of probs and labels do not match: {len(probs)} vs {len(labels)}")
            return 0

        return average_precision_score(labels, probs)

    def micro_f1_from_micro_scores(self, scores, labels):
        # Convert scores to binary predictions: 1 for [2, 3], 0 for [0, 2)
        predictions = np.array([1 if 2 <= score <= 3 else 0 for score in scores])
        labels = np.array(labels)

        # Check for mismatched lengths
        if len(predictions) != len(labels):
            print(f"Length of predictions and labels do not match: {len(predictions)} vs {len(labels)}")
            return 0

        # Calculate precision, recall, and F1 score
        tp = np.sum((predictions == 1) & (labels == 1))
        fp = np.sum((predictions == 1) & (labels == 0))
        fn = np.sum((predictions == 0) & (labels == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return f1

    def macro_f1_for_pos(self, pos_data):

        predictions = [1 if 2 <= score <= 3 else 0 for score in pos_data['avg_scores']]
        labels = pos_data['avg_binary_pid_rels']

        # Ensure predictions and labels have the same length
        if len(predictions) != len(labels):
            print(f"Length mismatch: predictions ({len(predictions)}) and labels ({len(labels)})")
            return 0

        # Calculate precision, recall, and F1 score for each class
        tp = sum((p == 1 and l == 1) for p, l in zip(predictions, labels))
        fp = sum((p == 1 and l == 0) for p, l in zip(predictions, labels))
        fn = sum((p == 0 and l == 1) for p, l in zip(predictions, labels))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return f1

    def macro_mse_for_pos(self, pos_data):
        preds = np.clip(np.array(pos_data['avg_scores']), 0, 3)
        labels = []
        for label in pos_data['avg_binary_pid_rels']:
            if label not in [0, 1]:
                return 0
            labels.append(label * 3)
        labels = np.array(labels)

        # Ensure predictions and labels have the same length
        if len(preds) != len(labels):
            print(f"Length mismatch: predictions ({len(preds)}) and labels ({len(labels)})")
            return 0

        # Calculate Mean Squared Error
        mse = np.mean((labels - preds) ** 2)
        return mse

    def macro_auc_for_pos(self, pos_data):
        # Convert average scores to probabilities (0-3 scaled to 0-1)
        probs = np.clip(np.array(pos_data['avg_scores']), 0, 3) / 3.0
        labels = np.array(pos_data['avg_binary_pid_rels'])

        for label in labels:
            if label not in [0, 1]:
                return 0

        # Ensure probabilities and labels have the same length
        if len(probs) != len(labels):
            print(f"Length mismatch: probabilities ({len(probs)}) and labels ({len(labels)})")
            return 0

        # Calculate AUC-PR
        return average_precision_score(labels, probs)

    def parse_llm_list(self, llm_output):
        # Try parsing the LLM output using JSON list reader
        try:
            return json.loads(llm_output)
        except json.JSONDecodeError:
            #self.logger.debug(f"Could not parse LLM output as list using regex parsing to look for list in LLM output: {llm_output}")
            try:
                match = re.search(r'\[.*?\]', llm_output, re.DOTALL)
                if match:
                    # Extract and convert single-quoted strings to double quotes for JSON compatibility
                    extracted_list = match.group(0).replace("'", '"')
                    return json.loads(extracted_list)
                else:
                    self.logger.warning(f"No valid regex list found in LLM output: {llm_output}")
                    return []
            except Exception as e:
                self.logger.warning(f"Regex extraction failed to parse as JSON: {e}")
                return []
        except Exception as e:
            self.logger.warning(f"No valid list found in LLM output: {e}")
            return []
        

warnings.filterwarnings("ignore", message="No positive class found in y_true, recall is set to one for all thresholds.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate an experiment based on a config file.")
    parser.add_argument("-e", "--eval-dir", type=str, help="The path to the evaluation dir containing eval_config.yaml")
    parser.add_argument("--skip-existing", action="store_true", help="Skip evaluation if output files already exist.")
    args = parser.parse_args()

    eval_manager = EvalManager(args.eval_dir, skip_existing=args.skip_existing)
    eval_manager.evaluate_experiment()