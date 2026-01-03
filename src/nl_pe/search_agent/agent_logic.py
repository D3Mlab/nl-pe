from nl_pe.utils.setup_logging import setup_logging
import copy

#class with some basic agent actions
class AgentLogic():

    def __init__(self, config):
        self.config = config
        self.data_config = self.config.get('data')
        self.logger = setup_logging(self.__class__.__name__, self.config)
        self.logger.debug("Initializing AgentLogic with config")
        self.logger.debug(f"Data config: {self.data_config}")

    def gt_rel_oracle(self, state):
        self.logger.debug(f"State keys: {list(state.keys())}")
        pid_list = state['current_batch']
        self.logger.debug(f"Current batch pid_list head: {pid_list[:10]}")
        qrels_path = self.data_config.get('qrels_path')
        self.logger.debug(f"Qrels path: {qrels_path}")

        if not qrels_path:
            self.logger.error("Qrels path not specified in data config")
            raise ValueError("Qrels path not specified in data config")

        if not hasattr(self, 'qrels_map'):
            self.logger.debug("Loading qrels from file")
            self.qrels_map = {}
            with open(qrels_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        qid, _, pid, rel = parts[0], parts[1], parts[2], parts[3]
                        rel = float(rel)
                        if qid not in self.qrels_map:
                            self.qrels_map[qid] = {}
                        self.qrels_map[qid][pid] = rel
            self.logger.debug(f"Loaded qrels for {len(self.qrels_map)} queries")

        qid = str(state['qid']) 
        self.logger.debug(f"Query ID: {qid}")

        rel_pids_for_query = set(self.qrels_map.get(qid, {}).keys())

        intersection = [pid for pid in pid_list if pid in rel_pids_for_query]

        self.logger.debug(f"Relevant docs retrieved for qid {qid}: {intersection[:20]}")
        self.logger.debug(f"Total relevant for qid {qid}: {len(rel_pids_for_query)}")

        # Get relevance scores for the pid_list in the same order
        scores = [self.qrels_map.get(qid, {}).get(pid, 0) for pid in pid_list]
        self.logger.debug(f"Relevance scores for batch head: {scores[:100]}")

        # Ensure pid_to_score_dict exists in state
        if "pid_to_score_dict" not in state:
            state["pid_to_score_dict"] = {}

        for pid in pid_list:
            if pid not in state["pid_to_score_dict"]:
                state["pid_to_score_dict"][pid] = []
        # Extend the scores for the pids in the batch
        for pid, score in zip(pid_list, scores):
            state["pid_to_score_dict"][pid].append(score)
        self.logger.debug(f"Updated pid_to_score_dict with {len(scores)} scores")

    def agg_pointwise_scores(self, state):
        self.logger.debug(f"Number of pids in pid_to_score_dict: {len(state['pid_to_score_dict'])}")

        state['pid_to_agg_score_dict'] = {}
        for pid, scores in state["pid_to_score_dict"].items():
            avg_score = sum(scores) / len(scores) if scores else 0
            state['pid_to_agg_score_dict'][pid] = avg_score

        # Create index dict for tie-breaking using init_knn_pid_list order
        index_dict = {pid: idx for idx, pid in enumerate(state['init_knn_pid_list'])}
        self.logger.debug(f"Created index dict for {len(index_dict)} pids")

        scored_pids = state['pid_to_agg_score_dict'].keys()
        sorted_pids = sorted(scored_pids, key=lambda pid: (-state['pid_to_agg_score_dict'].get(pid, 0), index_dict.get(pid, float('inf'))))
        self.logger.debug(f"Sorted top_k_psgs: {sorted_pids[:5]}...")  # Log first 5 for brevity
        state['top_k_psgs'] = sorted_pids
        state['top_k_rel_scores'] = [state['pid_to_agg_score_dict'].get(pid) for pid in sorted_pids]
        self.logger.debug(f"Top k rel scores: {state['top_k_rel_scores'][:5]}...")

    #START Batching methods for selecting passages for relevance judgments
    #################################################################
    def batch_all_dense(self, state):
        if not self.config.get('rel_batching'):
            state['current_batch'] = state['top_k_psgs']
        else:
            self.logger.error("rel batching config not implemented yet")
            self.logger.debug("Config indicates rel batching, but not implemented")
    #END Batching methods for selecting passages for relevance judgments
    #################################################################
