import os
import json

def get_mean_at_k(dir_path, metric, k):
    """
    Reads `all_queries_trec_eval_results.jsonl` from `dir_path`
    and returns the value of `mean_{metric}_{k}`.
    Example key: mean_P_1, mean_ndcg_10, etc.
    """
    file_path = os.path.join(dir_path, "all_queries_trec_eval_results.jsonl")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} does not exist.")

    key = f"mean_{metric}_{k}"

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)  # treat as single JSON object

    if key not in data:
        raise KeyError(f"{key} not found in {file_path}.")

    return data[key]