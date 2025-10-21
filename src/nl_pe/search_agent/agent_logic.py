from nl_pe.utils.setup_logging import setup_logging
import copy

#class with some basic agent actions
class AgentLogic():

    def __init__(self, config):
        self.config = config
        self.data_config = self.config.get('data')
        self.logger = setup_logging(self.__class__.__name__, self.config)

    def gt_rel_oracle(self, state):

        pid_list = state['current_batch']
        qrels_path = self.data_config.get('qrels_path')
        #standard qrels.txt format <qid 0 docid rel>
        #for each passage_id in state['current_batch'], get the relevance label from qrels

        # Load qrels if not already loaded (to avoid reloading for performance, but since small, could load each time)
        if not hasattr(self, 'qrels_map'):
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

        qid = state['qid']  # Assuming qid is in state

        # Get relevance scores for the pid_list in the same order
        scores = [self.qrels_map.get(qid, {}).get(pid, 0) for pid in pid_list]

        # Ensure pid_to_score_dict exists in state
        if "pid_to_score_dict" not in state:
            state["pid_to_score_dict"] = {}

        for pid in pid_list:
            if pid not in state["pid_to_score_dict"]:
                state["pid_to_score_dict"][pid] = []

        # Extend the scores for the pids in the batch
        for pid, score in zip(pid_list, scores):
            state["pid_to_score_dict"][pid].append(score)

    def agg_pointwise_scores(self, state):

        state['pid_to_agg_score_dict'] = {}
        for pid, scores in state["pid_to_score_dict"].items():
            avg_score = sum(scores) / len(scores) if scores else 0
            state['pid_to_agg_score_dict'][pid] = avg_score

        # Create index dict for tie-breaking using init_knn_pid_list order
        index_dict = {pid: idx for idx, pid in enumerate(state['init_knn_pid_list'])}
        
        scored_pids = state['pid_to_agg_score_dict'].keys()
        sorted_pids = sorted(scored_pids, key=lambda pid: (-state['pid_to_agg_score_dict'].get(pid, 0), index_dict.get(pid, float('inf'))))
        state['top_k_psgs'] = sorted_pids
        state['top_k_rel_scores'] = [state['pid_to_agg_score_dict'].get(pid) for pid in sorted_pids]

    #START Batching methods for selecting passages for relevance judgments
    #################################################################
    def batch_all_dense(self, state):
        if not self.config.get('rel_batching'):
            state['current_batch'] = state['top_k_psgs']
        self.logger.error("rel batching config not implemented yet")

    #END Batching methods for selecting passages for relevance judgments
    #################################################################
