import json
from nl_pe.utils.setup_logging import setup_logging

class CEScorer():

    def __init__(self,config):
        self.config = config
        self.logger = setup_logging(self.__class__.__name__, self.config)


        #load model

    def score(self,state,doc_ids):
        
        #last component of cache path is qid

        #try to look up did result and times
            #return if success
        #normalize
        #write to cache  
        pass
