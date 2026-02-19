import json
from nl_pe.utils.setup_logging import setup_logging
from nl_pe.utils.scorer import Scorer, TextScorer
from sentence_transformers import CrossEncoder
from nl_pe.embedding.embedders import normalize_device
import torch
import os
import time
import numpy as np

class CEScorer(TextScorer):

    def __init__(self,config):
        super().__init__(config)

        self.device = normalize_device(self.config.get('inference_device')) #'gpu' or 'cpu' in config
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
        self.batch_size = self.config.get('data',{}).get('embedding_batch_size')

    def score(self,state,doc_ids):
        batch_size = min(self.batch_size, len(doc_ids))

        self.logger.debug(
            "CEScorer.score called | qid=%s | n_doc_ids=%d | batch_size=%d",
            state.get("qid", "unknown"),
            len(doc_ids),
            batch_size,
        )

        scores = []
        times = []
        #track what is not cached
        missing_doc_ids = []

        # --------------------------------------------------
        # 1. Check cache
        # --------------------------------------------------
        for did in doc_ids:
            key = f"{did}::bs::{batch_size}"
            if key in self.cache:
                entry = self.cache[key]
                scores.append(float(entry["score"]))
                times.append(float(entry["time"]))
            else:
                missing_doc_ids.append(did)

        self.logger.debug(
            "Cache check done | hits=%d | misses=%d",
            len(doc_ids) - len(missing_doc_ids),
            len(missing_doc_ids),
        )

        # If ALL present → return immediately
        if len(missing_doc_ids) == 0:
            self.logger.debug("All scores found in cache; skipping CE inference")
            state["observation_times"] += times
            return scores

        d_texts = self.did_to_text(state, doc_ids)
        query_text = state.get("query")

        pairs = [(query_text, d_text) for d_text in d_texts]

        self.logger.debug(
            "Running CE inference | n_pairs=%d | query_len=%d",
            len(pairs),
            len(query_text) if query_text is not None else -1,
        )

        start_t = time.time()

        with torch.no_grad():
            scores = self.model.predict(
            pairs,
            batch_size=batch_size,
            convert_to_numpy=True
        )
        inf_time = time.time() - start_t
        scores = scores.tolist() #detach from gpu

        self.logger.debug(
            "CE inference complete | n_scores=%d | inf_time=%.4fs",
            len(scores),
            inf_time,
        )

        # --------------------------------------------------
        # 4. Cache missing entries only
        # --------------------------------------------------
        per_doc_time = inf_time / len(doc_ids)

        for did, score in zip(doc_ids, scores):
            key = f"{did}::bs::{batch_size}"
            if key not in self.cache:
                self.cache[key] = {
                    "score": float(score),
                    "time": float(per_doc_time)
                }

        self.logger.debug(
            "Cache updated for missing docs | added=%d | per_doc_time=%.6fs",
            len(missing_doc_ids),
            per_doc_time,
        )

        state["observation_times"] += [inf_time]

        return scores 
    



