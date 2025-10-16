from pathlib import Path
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# corpus embeddings
def plot_embedding_times(method_paths, method_names, styles=None, title="Embedding Time per Method"):
    embedding_times = []
    valid_methods = []

    print("Loading embedding times...")

    for i, path in enumerate(method_paths):
        path = Path(path)  # convert to Path object
        file_path = path / "detailed_results.json"
        if not file_path.exists():
            print(f"Warning: {file_path} does not exist. Skipping.")
            continue

        with open(file_path, "r") as f:
            data = json.load(f)
            embedding_time = data.get("embedding_time")
            if embedding_time is None:
                print(f"Warning: 'embedding_time' not found in {file_path}. Skipping.")
                continue
            embedding_times.append(embedding_time)
            valid_methods.append(method_names[i])

    if not embedding_times:
        print("No embedding times found. Exiting.")
        return

    # Set up bar styles
    if styles is None:
        styles = [{}] * len(embedding_times)
    elif len(styles) < len(embedding_times):
        styles += [styles[-1]] * (len(embedding_times) - len(styles))

    # Plot
    fig, ax = plt.subplots(figsize=(8,6))
    bars = ax.bar(valid_methods, embedding_times)

    # Apply styles
    for bar, style in zip(bars, styles):
        if "color" in style:
            bar.set_color(style["color"])
        if "hatch" in style:
            bar.set_hatch(style["hatch"])

    ax.set_ylabel("Embedding Time (s)")
    ax.set_title(title)
    ax.grid(axis="y")
    plt.tight_layout()
    plt.show()


def plot_knn_times_scatter(method_paths, method_names, styles=None, title="KNN Time per Method"):
    knn_times_list = []
    valid_methods = []

    print("Loading KNN times...")

    for i, path in enumerate(method_paths):
        path = Path(path)
        file_path = path / "all_queries_knn_times.csv"
        if not file_path.exists():
            print(f"Warning: {file_path} does not exist. Skipping.")
            continue

        df = pd.read_csv(file_path)
        if "knn_time" not in df.columns:
            print(f"Warning: 'knn_time' column not found in {file_path}. Skipping.")
            continue

        knn_times_list.append(df["knn_time"].values)
        valid_methods.append(method_names[i])

    if not knn_times_list:
        print("No KNN times found. Exiting.")
        return

    # Set up scatter styles
    if styles is None:
        styles = [{}] * len(knn_times_list)
    elif len(styles) < len(knn_times_list):
        styles += [styles[-1]] * (len(knn_times_list) - len(styles))

    # Plot vertical scatter per method
    fig, ax = plt.subplots(figsize=(10,6))
    for i, (times, method, style) in enumerate(zip(knn_times_list, valid_methods, styles)):
        x = np.full_like(times, i)  # all points at the method's x-position
        kwargs = {}
        if "color" in style:
            kwargs["color"] = style["color"]
        if "marker" in style:
            kwargs["marker"] = style["marker"]
        ax.scatter(x, times, label=method, **kwargs)

    ax.set_xticks(range(len(valid_methods)))
    ax.set_xticklabels(valid_methods)
    ax.set_ylabel("KNN Time (s)")
    ax.set_title(title)
    ax.grid(axis="y")
    plt.tight_layout()
    plt.show()


# -----------------------------
# 2. Boxplot per method (median/IQR/1.5*IQR whiskers)
# -----------------------------
def plot_knn_times_boxplot(method_paths, method_names, styles=None, title="KNN Time Distribution"):
    knn_times_list = []
    valid_methods = []

    print("Loading KNN times for boxplot...")

    for i, path in enumerate(method_paths):
        path = Path(path)
        file_path = path / "all_queries_knn_times.csv"
        if not file_path.exists():
            print(f"Warning: {file_path} does not exist. Skipping.")
            continue

        df = pd.read_csv(file_path)
        if "knn_time" not in df.columns:
            print(f"Warning: 'knn_time' column not found in {file_path}. Skipping.")
            continue

        knn_times_list.append(df["knn_time"].values)
        valid_methods.append(method_names[i])

    if not knn_times_list:
        print("No KNN times found. Exiting.")
        return

    # Plot boxplot
    fig, ax = plt.subplots(figsize=(8,6))
    bp = ax.boxplot(knn_times_list, labels=valid_methods, patch_artist=True,
                    showfliers=False, whis=1.5)

    # Apply colors if provided
    if styles:
        for patch, style in zip(bp['boxes'], styles):
            if "color" in style:
                patch.set_facecolor(style["color"])

    ax.set_ylabel("KNN Time (s)")
    ax.set_title(title)
    ax.grid(axis="y")
    plt.tight_layout()
    plt.show()
    