import json
import os
import math
from nl_pe.utils.setup_logging import setup_logging
import random
from copy import deepcopy
from math import prod
from collections import Counter
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd

#class with some basic agent actions
class AgentLogic():

    def __init__(self, config):
        self.config = config
        self.logger = setup_logging(self.__class__.__name__, self.config)
        self.batch_size = self.config['rerank'].get('batch_size')

    def get_next_batch(self,state):
        self.logger.debug('getting next batch')
        
        if not state.get('batch_queue'):
            self.logger.warning("Batch queue is unexpectedly empty. Setting current_batch to None.")
            state['current_batch'] = None

        state['current_batch'] = state['batch_queue'].pop(0)
    
    def lw_simple_postprocess(self, state):

        state['top_k_psgs'] = [{'pid': pid} for pid in state['valid_batch_pid_lists'][0]]

        self.lw_clean_final_state(state)

    def lw_rank_agg_postprocess(self, state):
        # make positional pids based on initial list, 
        # e.g. first passage in initial list gets pos_pid = 0, etc
        init_psg_list = state['instance']['psg_list']
        pos_pids = {psg['pid']: idx for idx, psg in enumerate(init_psg_list)}

        # Map valid_batch_pid_lists pids (strings) to positional pids
        valid_batch_pos_pid_lists = np.array([
            [pos_pids[pid] for pid in batch] for batch in state['valid_batch_pid_lists']
        ])

        if self.config['rerank']['lw_rank_agg_func'] == 'kemeny_young':
            try:
                agg_result = KemenyOptimalAggregator().aggregate(valid_batch_pos_pid_lists)
                if agg_result:
                    agged_prefs = agg_result['solution']
                    state['agg_time'] = agg_result['solution_time']
                else:
                    self.logger.error("KemenyOptimalAggregator returned empty preferences.")
            except:
                self.logger.error("KemenyOptimalAggregator failed to aggregate preferences.")
        else:
            raise ValueError("missing or invalid ['rerank']['lw_rank_agg_func']")

        state['top_k_psgs'] = [{'pid': init_psg_list[pos_pid]['pid']} for pos_pid in agged_prefs] 

        self.lw_clean_final_state(state)

    def pw_postprocess(self, state):
        
        #aggregate multiple scores for each passage
        score_agg_func = getattr(self, self.config['agent']['score_agg_func'])
        score_agg_func(state)

        pid_to_agg_score_dict = state["pid_to_agg_score_dict"]

        # Create top_k_psgs sorted by aggregate score and initial order
        sorted_pids = sorted(
            pid_to_agg_score_dict,
            key=lambda pid: (-pid_to_agg_score_dict[pid], state['instance']['psg_list'].index(next(psg for psg in state['instance']['psg_list'] if psg['pid'] == pid)))
        )
        state['top_k_psgs'] = [{'pid': pid} for pid in sorted_pids]

        # Create scores list corresponding to top_k_psgs
        state['scores'] = [pid_to_agg_score_dict[pid] for pid in sorted_pids]

        # Create scores_init_order list corresponding to the initial order of psg_list
        # None when pid has no scores (e.g. hallucinations)
        state['scores_init_order'] = [pid_to_agg_score_dict.get(psg['pid']) for psg in state['instance']['psg_list']]

        #remove text from instance
        for p in state['instance']['psg_list']:
            p.pop('text', None)

        #del current batch from state if it exists
        state.pop('current_batch', None)

        #end rank() process
        state['terminate'] = True

    def preprocess_batch_sequential(self, state):
        #based on the state['instance']['psg_list'] which has the form
        #[{pid: <pid>, text: <text>}, {...},...], create a queue state['batch_queue'] where each item in the queue is a lists of dicts of the same form 
        # (i.e. [{pid: <pid>, text: <text>}, {...},...]) by slicing the passage list into batches of size self.batch_size, possibly with a smaller batch at the end (if the psg list doesn't split evenly into batches)
        psg_list = state['instance']['psg_list']
        batch_queue = []

        n_rescores = self.config['rerank'].get('n_rescores')

        for i in range(0, len(psg_list), self.batch_size):
            for j in range(n_rescores):
                batch_queue.append(psg_list[i:i + self.batch_size])

        self.create_batch_queue(state, batch_queue)

    def preprocess_batch_sequential_in_batch_shuffle(self, state):
        #based on the state['instance']['psg_list'] which has the form
        #[{pid: <pid>, text: <text>}, {...},...], create a queue state['batch_queue'] where each item in the queue is a lists of dicts of the same form 
        # (i.e. [{pid: <pid>, text: <text>}, {...},...]) by slicing the passage list into batches of size self.batch_size, possibly with a smaller batch at the end (if the psg list doesn't split evenly into batches)
        #then setting the random seed to self.config['random_seed'],
        #creating self.config['rerank']['n_shuffles'] number of shuffles for each batch, and INSTEAD of the initial order, creating n_shuffles versions of each batch
        #which are added to the queue which is flat and looks like [batch1_shuffle1, batch1_shuffle2, ... batch1_shufflen, batch2_shuffle1, ...]
        psg_list = state['instance']['psg_list']
        batch_queue = []
        random_seed = self.config.get('random_seed', 42)
        random.seed(random_seed)
        n_shuffles = self.config['rerank'].get('n_rescores')

        for i in range(0, len(psg_list), self.batch_size):
            batch = psg_list[i:i + self.batch_size]
            for _ in range(n_shuffles):
                shuffled_batch = batch.copy()
                random.shuffle(shuffled_batch)
                batch_queue.append(shuffled_batch)

        self.create_batch_queue(state, batch_queue)

    def preprocess_batch_inter_batch_shuffle(self, state):
        psg_list = deepcopy(state['instance']['psg_list'])
        batch_queue = []
        random_seed = self.config.get('random_seed', 42)
        random.seed(random_seed)
        n_shuffles = self.config['rerank'].get('n_rescores')

        for i in range(n_shuffles):
            random.shuffle(psg_list)
            for j in range(0, len(psg_list), self.batch_size):
                batch_queue.append(psg_list[j:j + self.batch_size])

        self.create_batch_queue(state, batch_queue)

    def create_batch_queue(self, state, batch_queue):
        state['batch_queue'] = batch_queue
        state['batch_pid_history'] = [[item['pid'] for item in batch] for batch in batch_queue]
        state['preprocessing_done'] = True


    def increment_batch_indices(self, state):
        #deprecated?
        passage_list_length = len(state['instance']['psg_list'])

        # Check if the batch indices are not set    
        if not (state.get('batch_start_idx') or state.get('batch_end_idx')):
            state['batch_start_idx'] = 0
            state['batch_end_idx'] = self.batch_size
        else:
            state['batch_start_idx'] += self.batch_size
            state['batch_end_idx'] += self.batch_size

        # Check if the batch indices go past the length of the passage list
        if state['batch_start_idx'] >= passage_list_length:
            self.logger.error("Batch start index is beyond the passage list length.")
            state['terminate'] = True
            return state

        # Adjust the batch end index if it goes past the length of the passage list
        if state['batch_end_idx'] >= passage_list_length:
            state['batch_end_idx'] = passage_list_length

        return state

    def amean(self,state):
        pid_to_score_dict = state.get("pid_to_score_dict", {})
        pid_to_agg_score_dict = {}

        for pid, scores in pid_to_score_dict.items():
            if scores: #for non empty score lists
                avg_score = sum(scores) / len(scores)
            else: #for empty score lists
                avg_score = 0
            pid_to_agg_score_dict[pid] = avg_score

        state["pid_to_agg_score_dict"] = pid_to_agg_score_dict

    def hmean(self,state):
        pid_to_score_dict = state.get("pid_to_score_dict", {})
        pid_to_agg_score_dict = {}

        for pid, scores in pid_to_score_dict.items():
            if scores:  # For non-empty score lists
                if all(score > 0 for score in scores):  # Harmonic mean requires positive scores
                    harmonic_mean = len(scores) / sum(1 / score for score in scores)
                    pid_to_agg_score_dict[pid] = harmonic_mean
                else:
                    pid_to_agg_score_dict[pid] = 0  # Handle non-positive scores gracefully
            else:  # For empty score lists
                pid_to_agg_score_dict[pid] = 0  
        state["pid_to_agg_score_dict"] = pid_to_agg_score_dict

    def gmean(self,state):
        pid_to_score_dict = state.get("pid_to_score_dict", {})
        pid_to_agg_score_dict = {}

        for pid, scores in pid_to_score_dict.items():
            if scores:
                if all(score > 0 for score in scores):  # Geometric mean requires positive scores
                    geometric_mean = prod(scores) ** (1 / len(scores))
                    pid_to_agg_score_dict[pid] = geometric_mean
                else:
                    pid_to_agg_score_dict[pid] = 0  # Handle non-positive scores gracefully
            else:  # For empty score lists
                pid_to_agg_score_dict[pid] = 0       

        state["pid_to_agg_score_dict"] = pid_to_agg_score_dict

    def maxscore(self,state):
        pid_to_score_dict = state.get("pid_to_score_dict", {})
        pid_to_agg_score_dict = {}

        for pid, scores in pid_to_score_dict.items():
            if scores:
                pid_to_agg_score_dict[pid] = max(scores)
            else:  # For empty score lists
                pid_to_agg_score_dict[pid] = 0       

        state["pid_to_agg_score_dict"] = pid_to_agg_score_dict

    def minscore(self,state):
        pid_to_score_dict = state.get("pid_to_score_dict", {})
        pid_to_agg_score_dict = {}

        for pid, scores in pid_to_score_dict.items():
            if scores:
                pid_to_agg_score_dict[pid] = min(scores)
            else:
                pid_to_agg_score_dict[pid] = 0

        state["pid_to_agg_score_dict"] = pid_to_agg_score_dict

    def majvote(self,state):
        pid_to_score_dict = state.get("pid_to_score_dict", {})
        pid_to_agg_score_dict = {}

        for pid, scores in pid_to_score_dict.items():
            if scores:
                most_common_score = Counter(scores).most_common(1)[0][0]
                pid_to_agg_score_dict[pid] = most_common_score
            else:
                pid_to_agg_score_dict[pid] = 0  # Handle empty scores gracefully

        state["pid_to_agg_score_dict"] = pid_to_agg_score_dict

    def lw_clean_final_state(self,state): 
    #remove text from instance
        for p in state['instance']['psg_list']:
            p.pop('text', None)

        #del current batch from state if it exists
        state.pop('current_batch', None)

        #end rank() process
        state['terminate'] = True

#linear regression
def lr(x,y):
    x = np.array(x).reshape(-1, 1)
    y = np.array(y)

    model = LinearRegression()
    model.fit(x, y)

    lr_slope = model.coef_[0]
    lr_intercept = model.intercept_

    return lr_slope, lr_intercept

#other misc helper functions
def get_doc_text_list(ids, corpus_path):
    #ids: list of doc_ids
    #return [{"docID": d1, "text": <text_d1>},...]

    if not ids:
        return []

    id_set = set(ids)
    id_and_text_dict = {}

    # Iterate through the corpus and collect only the required documents
    for doc in jsonl_line_generator(corpus_path):
        doc_id = doc.get('docID')
        if doc_id in id_set:
            id_and_text_dict[doc_id] = doc.get('text')
            if len(id_and_text_dict) == len(ids):
                break

    id_and_text_list = [{'docID': doc_id, 'text': id_and_text_dict[doc_id]} for doc_id in ids if doc_id in id_and_text_dict]

    return id_and_text_list

def jsonl_line_generator(path):
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                yield json.loads(line)
