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
        self.data_config.get('qrels_path')
        #standard qrels.txt format <qid 0 docid rel>
        #for each passage_id in state['current_batch'], get the relevance label from qrels as a list of scores in the same order as pid_list

        # Ensure pid_to_score_dict exists in state
        if "pid_to_score_dict" not in state:
            state["pid_to_score_dict"] = {}

        for pid in pid_list:
            if pid not in state["pid_to_score_dict"]:
                state["pid_to_score_dict"][pid] = []    

        # Extend the scores for the pids in the batch
        for pid, score in zip(pid_list, scores):
            state["pid_to_score_dict"][pid].append(score)

    #START Batching methods for selecting passages for relevance judgments
    #################################################################
    def batch_all_dense(self, state):
        if not self.config.get('rel_batching'):
            state['current_batch'] = state['top_k_psgs']
        self.logger.error("rel batching config not implemented yet")

    #END Batching methods for selecting passages for relevance judgments
    #################################################################

        
