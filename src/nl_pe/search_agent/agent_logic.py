from nl_pe.utils.setup_logging import setup_logging
import copy

#class with some basic agent actions
class AgentLogic():

    def __init__(self, config):
        self.config = config
        self.data_config = self.config.get('data')
        self.logger = setup_logging(self.__class__.__name__, self.config)

    def gt_rel_oracle(self, state):
        
        self.data_config.get('qrels_path')
        #standard qrels.txt format <qid 0 docid rel>
        #for each passage_id in state['current_batch'], get the relevance label from qrels



    #START Batching methods for selecting passages for relevance judgments
    #################################################################
    def batch_all_dense(self, state):
        if not self.config.get('rel_batching'):
            state['current_batch'] = state['top_k_psgs']
        self.logger.error("rel batching config not implemented yet")

    #END Batching methods for selecting passages for relevance judgments
    #################################################################

        
