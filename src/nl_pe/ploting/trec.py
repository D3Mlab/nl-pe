import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def _normalize_metric_name(metric):
    return str(metric).strip().lower()


def _extract_trec_baselines_for_dataset(baselines, dataset):
    if baselines is None:
        return {}

    if not dataset:
        raise ValueError("When 'baselines' is provided, 'dataset' must also be provided.")

    supported_methods = {
        "bm25 flat": ("BM25 Flat", 0.30),
        "bm25 mf": ("BM25 MF", 0.66),
        "splade": ("SPLADE", 1.00),
    }

    dataset_key = str(dataset).strip().lower()
    extracted = {}

    if isinstance(baselines, pd.DataFrame):
        cols = {str(c).strip().lower(): c for c in baselines.columns}
        dataset_col = cols.get("dataset")
        method_col = cols.get("method")
        ndcg_col = cols.get("ndcg@10") or cols.get("ndcg_10") or cols.get("ndcg_cut_10") or cols.get("ndcg10")
        recall_col = cols.get("recall@100") or cols.get("recall_100") or cols.get("recall100")

        if not all([dataset_col, method_col, ndcg_col, recall_col]):
            print(
                "Warning: baselines DataFrame is missing required columns "
                "(dataset, method, ndcg@10, recall@100). Skipping baseline markers."
            )
            return {}

        for _, row in baselines.iterrows():
            if str(row[dataset_col]).strip().lower() != dataset_key:
                continue

            method_key = str(row[method_col]).strip().lower()
            if method_key not in supported_methods:
                continue

            method_name, alpha = supported_methods[method_key]
            extracted[method_name] = {
                "ndcg": float(row[ndcg_col]),
                "recall": float(row[recall_col]),
                "alpha": alpha,
            }

        return extracted

    for item in baselines:
        if isinstance(item, (list, tuple)) and len(item) >= 4:
            row_dataset, row_method, ndcg_10, recall_100 = item[:4]
        elif isinstance(item, dict):
            row_dataset = item.get("dataset")
            row_method = item.get("method")
            ndcg_10 = item.get("ndcg@10", item.get("ndcg_10", item.get("ndcg_cut_10")))
            recall_100 = item.get("recall@100", item.get("recall_100"))
        else:
            continue

        if str(row_dataset).strip().lower() != dataset_key:
            continue

        method_key = str(row_method).strip().lower()
        if method_key not in supported_methods:
            continue

        method_name, alpha = supported_methods[method_key]
        extracted[method_name] = {
            "ndcg": float(ndcg_10),
            "recall": float(recall_100),
            "alpha": alpha,
        }

    return extracted


def _plot_trec_baseline_markers(ax, metric, k, baseline_points):
    metric_name = _normalize_metric_name(metric)
    if metric_name == "ndcg":
        x_value, y_key = 10, "ndcg"
    elif metric_name == "recall":
        x_value, y_key = 100, "recall"
    else:
        return

    if x_value > k:
        return

    for baseline in baseline_points.values():
        ax.scatter(
            x_value,
            baseline[y_key],
            marker="x",
            color="black",
            alpha=baseline["alpha"],
            s=60,
            zorder=10,
        )

def plot_trec_metrics_vs_k(step_size, k, metrics, method_paths, method_names,
                 line_styles=None, y_mins=None, y_maxs=None, title=None, dataset = '', baselines=None):
    if k % step_size != 0:
        print('k must be divisible by step size')
        return

    print('Loading data...')

    df = get_trec_df_method_set(method_paths, method_names)

    print("Columns:", df.columns.tolist())
    print("Shape:", df.shape)
    print("First row:", df.head(1).to_dict(orient="records"))

    num_metrics = len(metrics)
    baseline_points = _extract_trec_baselines_for_dataset(baselines, dataset)

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

        _plot_trec_baseline_markers(ax, metric, k, baseline_points)
        
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


