import json
from nl_pe.utils.setup_logging import setup_logging
from nl_pe.utils.scorer import Scorer
from sentence_transformers import CrossEncoder
from nl_pe.embedding.embedders import normalize_device
import torch
import os

class CEScorer(Scorer):

    def __init__(self,config):
        super().__init__(config)
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

        self.normalize_scores = self.config.get('observation', {}).get('normalize_scores', False)
        self.batch_size = self.config.get('data',{}).get('embedding_batch_size')

    
    def score(self,state,doc_ids):
        batch_size = min(self.batch_size, len(doc_ids))

        scores = []
        times = []

        #check if ALL doc_ids already have cached scores
        for did in doc_ids:
            key = f"{did}::bs::{batch_size}"
            if key in self.cache:
                entry = self.cache[key]
                score = entry["score"]
                time = entry["time"]
                scores.append(score)
                times.append(time)
            else:
                scores = []
                times = []
                break
        
        if len(scores) > 0:
            state['observation_times'] += times
            return scores
        
        #if at least one doc id is missing values, recompute everything (but dont overwrite existing)
        #get list of doc_texts from corpus json

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