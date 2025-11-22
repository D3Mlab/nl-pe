from abc import ABC
from nl_pe.utils.setup_logging import setup_logging
import os

class BaseActiveLearner(ABC):

    def __init__(self, config):

        self.config = config
        self.logger = setup_logging(self.__class__.__name__, config = self.config, output_file=os.path.join(self.config['exp_dir'], "experiment.log"))
        self.n_obs_iterations = self.config.get('active_learning', {}).get('n_obs_iterations')

    def get_single_rel_judgment(self, state, doc_id):
        #for now, we just use the qrels oracle, later we will potenitally use an llm, add noise, etc.
        #modify this agent_logic code to just get the relevance label of doc_id for the current qid in state, it should be much simpler!!!

        #  self.logger.debug(f"State keys: {list(state.keys())}")
        # pid_list = state['current_batch']
        # self.logger.debug(f"Current batch pid_list head: {pid_list[:10]}")
        # qrels_path = self.data_config.get('qrels_path')
        # self.logger.debug(f"Qrels path: {qrels_path}")

        # if not qrels_path:
        #     self.logger.error("Qrels path not specified in data config")
        #     raise ValueError("Qrels path not specified in data config")

        # if not hasattr(self, 'qrels_map'):
        #     self.logger.debug("Loading qrels from file")
        #     self.qrels_map = {}
        #     with open(qrels_path, 'r') as f:
        #         for line in f:
        #             parts = line.strip().split()
        #             if len(parts) >= 4:
        #                 qid, _, pid, rel = parts[0], parts[1], parts[2], parts[3]
        #                 rel = float(rel)
        #                 if qid not in self.qrels_map:
        #                     self.qrels_map[qid] = {}
        #                 self.qrels_map[qid][pid] = rel
        #     self.logger.debug(f"Loaded qrels for {len(self.qrels_map)} queries")

        # qid = state['qid'] 
        # self.logger.debug(f"Query ID: {qid}")

        # # Get relevance scores for the pid_list in the same order
        # scores = [self.qrels_map.get(qid, {}).get(pid, 0) for pid in pid_list]
        # self.logger.debug(f"Relevance scores for batch head: {scores[:10]}")

        # # Ensure pid_to_score_dict exists in state
        # if "pid_to_score_dict" not in state:
        #     state["pid_to_score_dict"] = {}

        # for pid in pid_list:
        #     if pid not in state["pid_to_score_dict"]:
        #         state["pid_to_score_dict"][pid] = []
        # # Extend the scores for the pids in the batch
        # for pid, score in zip(pid_list, scores):
        #     state["pid_to_score_dict"][pid].append(score)
        # self.logger.debug(f"Updated pid_to_score_dict with {len(scores)} scores")
        # pass

    def final_ranked_list_from_posterior(self, state):
        #todo
        #in the state, which is the final ranked list of doc ids based on the posterior means in the last iteration (state["posterior_means"][-1])
        pass

class GPActiveLearner(BaseActiveLearner):

    def __init__(self, config):
        super().__init__(config)

    def active_learn(self, state):
        pass
        #initialize gp with variables read from config, e.g.:
# ...
# gp:
#   kernel: rbf
#   lengthscale: 1
#   signal_noise: 1
#   observation_noise: 0.1
#   query_rel_label: 2

        #the first observation we will make is the query embedding (i.e. state["query_emb"]) and the query relevance label (read from config)


        #now that we have our first observations, 
        # we will iterate n_obs_iterations times to select new points to observe and update the GP
        # in each iteration we need to decide on the next point to observe, which is determined by our acquisition function (we will pick a doc id which corresponds to an embedding... notet that we will never load all embeddings into memory) read from config:
# ...
# active_learning:
#   n_obs_iterations: 100
#   aquisition_f: ts #could also be ucb, greedy, greedy_epsilon, random
        #note the methods below that we have for these aquisition functions

        #from the each aquisition step, we will record the selected doc id in a list state["selected_doc_ids"]
        #we will also record a list of aquisition scores for each iteration in state["acquisition_scores"]
        #we will also record aquisition times for each iteration in state["acquisition_times"]

        #then we need to update the GP with the new observation (embedding and relevance label)
        #for now we are assuming fixed hyperparameters for the gp, but in future we may want to optimize them after each update
        #after each update, we will record the mean and variance of the GP over all doc_ids (observed and unobserved), as two-level lists state["posterior_means"] and state["posterior_variances"] where the outer list is over iterations and the inner list is over doc_ids

        #doc_ids always follow the order of the embedding index and can be obtainted from config['data']:
# data:
#   d_text_csv: data/ir/beir/nfcorpus/docs.csv
#   q_text_csv: data/ir/beir/nfcorpus/test_queries.csv
#   index_path: data/ir/beir/nfcorpus/miniLM/faiss/index
#   doc_ids_path: data/ir/beir/nfcorpus/miniLM/faiss/index_doc_ids.pkl
#   qrels_path: data\ir\beir\nfcorpus\qrels\test.txt


    #reminder: we don't load all embeddings into memory
    def ts(self, state):
        #Thompson Sampling acquisition function implementation
        pass
    def ucb(self, state):
        #Upper Confidence Bound acquisition function implementation
        pass
    def greedy(self, state):
        #Greedy acquisition function implementation
        pass
    def greedy_epsilon(self, state):
        #Greedy with epsilon exploration acquisition function implementation
        pass
    def random(self, state):
        #Random acquisition function implementation
        pass

