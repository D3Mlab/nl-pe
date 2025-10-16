import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def plot_trec_metrics_vs_k(step_size, k, metrics, method_paths, method_names,
                 line_styles=None, y_mins=None, y_maxs=None, title=None):
    if k % step_size != 0:
        print('k must be divisible by step size')
        return

    print('Loading data...')

    df = get_trec_df_method_set(method_paths, method_names)

    print("Columns:", df.columns.tolist())
    print("Shape:", df.shape)
    print("First row:", df.head(1).to_dict(orient="records"))

    num_metrics = len(metrics)

    fig, axes = plt.subplots(num_metrics, 1, figsize=(12, 6 * num_metrics))
    if num_metrics == 1:
        axes = [axes]  # make iterable for single metric

    for i, (metric, ax) in enumerate(zip(metrics, axes)):
        metric_values = get_trec_metric_values_from_method_df(df, k, metric)
        x_values = list(range(step_size, k + 1, step_size))
        
        for j, values in enumerate(metric_values):
            y_values = values[:k:step_size]
            if y_values is None or len(y_values) == 0:
                print(f"Warning: No data to plot for method '{method_names[j]}' on metric '{metric}'.")
                continue
            if line_styles and j < len(line_styles):
                ax.plot(x_values, y_values, label=method_names[j], **line_styles[j])
            else:
                ax.plot(x_values, y_values, label=method_names[j])
        
        if y_mins is not None and y_maxs is not None:
            ax.set_ylim(y_mins[i], y_maxs[i])
        else:
            ax.set_ylim(0, 1)
            
        ax.set_xlabel('K')
        ax.set_ylabel(f'{metric}@k')
        ax.set_title(f'{metric.upper()}@K')
        ax.grid(True)
        
        if i == 0:
            ax.legend()

    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if title and method_paths:
        save_path = os.path.join("plots", f"{title}.png")
        plt.savefig(save_path)

    plt.show()


def get_trec_df_method_set(method_paths, method_names):
    data = []

    for i,exp_dir in enumerate(method_paths):
        results_file = os.path.join(exp_dir, "all_queries_trec_eval_results.jsonl")
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

