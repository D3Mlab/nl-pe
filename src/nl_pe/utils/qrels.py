"""Utilities for loading qrels files in a shared, consistent way."""

from __future__ import annotations
from nl_pe.utils.setup_logging import setup_logging
from typing import Dict, TextIO
import os
import pytrec_eval


def load_qrels_map(qrels_path: str) -> Dict[str, Dict[str, float]]:
    """Load qrels into a nested dict: {qid: {doc_id: rel}}.

    Expects TREC qrels format with at least 4 columns: qid, _, doc_id, rel.
    """
    qrels_map: Dict[str, Dict[str, float]] = {}
    with open(qrels_path, "r") as qrels_file:
        for line in qrels_file:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            qid, _, doc_id, rel = parts[0], parts[1], parts[2], parts[3]
            qrels_map.setdefault(qid, {})[str(doc_id)] = float(rel)
    return qrels_map


def load_pytrec_eval_qrels(qrels_path: str) -> Dict[str, Dict[str, int]]:
    """Load qrels using pytrec_eval.parse_qrel for evaluation."""
    with open(qrels_path, "r") as qrels_file:
        return pytrec_eval.parse_qrel(qrels_file)
    
class GTScorer():

    def __init__(self,config):
        self.config = config
        self.logger = setup_logging(self.__class__.__name__, config = self.config, output_file=os.path.join(self.config['exp_dir'], "experiment.log"))
        self.logger.debug(f"Initializing {self.__class__.__name__} with config: {config}")

    def open_cache(self,qid,prompt_name = ''):
        data_config = self.config.get('data', {})
        qrels_path = data_config.get('qrels_path')
        if not qrels_path:
            self.logger.error("Qrels path not specified in data config")
            raise ValueError("Qrels path not specified in data config")
        self.qrels_map = load_qrels_map(qrels_path)
        self.logger.debug(f"Loaded qrels for {len(self.qrels_map)} queries")

    def score(self, state, doc_ids):
        doc_ids = [str(doc_id) for doc_id in doc_ids]
        self.logger.debug(
            "Getting relevance judgments for %d doc_ids with qid %s",
            len(doc_ids),
            state.get('qid', 'unknown'),
        )

        qid = str(state['qid'])
        rel_map = self.qrels_map.get(qid, {})
        judgments = [rel_map.get(doc_id, 0) for doc_id in doc_ids]
        self.logger.debug("Batch relevance judgments head: %s", judgments[:10])
        return judgments