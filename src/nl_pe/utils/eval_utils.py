#create dataframe with each experiment's trec results as a row
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_trec_df_method_set(method_paths, method_names):
    data = []

    for i,exp_dir in enumerate(method_paths):
        results_file = os.path.join(exp_dir, "all_queries_eval_results.jsonl")
        if os.path.exists(results_file):
            with open(results_file, "r") as file:
                results = json.load(file)
                results["experiment"] = method_names[i]
                data.append(results)
        else:
            print(f"Warning: {results_file} does not exist. Skipping this experiment.")

    df = pd.DataFrame(data)
        
    # Remove std columns
    df = df[[col for col in df.columns if not col.startswith("std_dev_")]]

    # Remove "mean_" prefix from column names
    df.columns = [col.replace("mean_", "") for col in df.columns]

    return df

#eg. metrics = ['ndcg', 'P', 'recall']
def get_trec_metric_values_from_method_df(df, k, metric):
    """
    Extracts metric values for a given k and metric from the DataFrame.
    """
    if metric in {'P', 'recall'}:
        metric_columns = [f"{metric}_{i}" for i in range(1, k+1)]
    else:
        metric_columns = [f"{metric}_cut_{i}" for i in range(1, k+1)]
    metric_values = df[metric_columns].values
    return [np.array(values) for values in metric_values]

def get_runtimes(method_path):
    runtimes_path = os.path.join(method_path, "all_total_prompt_runtimes.jsonl")
    
    if not os.path.exists(runtimes_path):
        return None
    
    with open(runtimes_path, "r") as file:
        runtimes_dict = json.load(file)
    
    runtimes = []
    for runtime in runtimes_dict.items():
        runtimes.append(runtime[1])

    runtimes = np.array(runtimes)
    return runtimes

