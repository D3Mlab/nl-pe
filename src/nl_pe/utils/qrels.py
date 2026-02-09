"""Utilities for loading qrels files in a shared, consistent way."""

from __future__ import annotations

from typing import Dict, TextIO

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