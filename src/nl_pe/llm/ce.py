import json
from nl_pe.utils.setup_logging import setup_logging
from sentence_transformers import CrossEncoder
from nl_pe.embedding.embedders import normalize_device
import torch

class CEScorer():

    def __init__(self,config):
        self.config = config
        self.logger = setup_logging(self.__class__.__name__, self.config)

        #device
        self.device = normalize_device(self.config.get('device').get('inference_device')) #'gpu' or 'cpu' in config
        use_fp16 = "cuda" in self.device
        self.logger.info(f"Loading CrossEncoder on device={self.device}")

        #model
        model_name = self.config.get('llm').get('model_name')
        self.model = CrossEncoder(
            model_name,
            device=self.device,
            automodel_args={"torch_dtype": torch.float16} if use_fp16 else {}
        )
        self.model.model.eval()

        self.cache_path = self.config.get('data',{}).get('cache_path')
        self.normalize_scores = bool(self.config.get('observation').get('normalize_scores'))

    def score(self,state,doc_ids):
        
        #last component of cache path is qid

        #try to look up did result and times
            #debug cache read
            #return if success
        #normalize
        #write to cache  
        #debug cache write


        #NOTE: could check performance without normalization of scores (example scores in api are -4, 8)

        pass



#in embedders:
# Device configuration: embedding operations use inference_device, tensor ops use tensor_ops_device
# def normalize_device(dev):
#     if dev == 'gpu' or dev == 'cuda':
#         return 'cuda:0'
#     elif dev == 'cpu':
#         return 'cpu'
#     else:
#         # Assume it's already a proper device string like 'cuda:1'
#         return dev