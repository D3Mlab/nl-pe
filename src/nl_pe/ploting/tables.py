import os
import json
import pandas as pd
import numpy as np

from typing import List


def make_two_metric_table(
    *,
    datasets: List[str],
    dataset_names: List[str],
    metric_str_l: str,
    metric_str_r: str,
    baseline_rows: List[str],
    method_rows: List[str],
    caption: str = "MAIN",
    label: str = "tab:main",
    table_env: str = "table*",
    size_cmd: str = r"\small",
) -> str:
    """
    Creates a LaTeX booktabs table with:
        - multirow Method column
        - two metrics per dataset
        - baselines block
        - methods block

    Assumes rows are already formatted as:
        "Method & val & val & ..."

    Returns full LaTeX string.
    """

    lines = []

    # ========================
    # TABLE START
    # ========================

    lines.extend([
        rf"\begin{{{table_env}}}[t]",
        size_cmd,
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
    ])

    col_spec = "l" + "cc" * len(datasets)
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    # ========================
    # HEADER
    # ========================

    # First row
    top_row = [r"\multirow{2}{*}{Method}"]
    for ds_name in dataset_names:
        top_row.append(rf"\multicolumn{{2}}{{c}}{{{ds_name}}}")

    lines.append(" & ".join(top_row) + r" \\")

    # cmidrules
    cmidrules = []
    start_col = 2
    for i in range(len(datasets)):
        left = start_col + i * 2
        right = left + 1
        cmidrules.append(rf"\cmidrule(lr){{{left}-{right}}}")

    lines.append(" ".join(cmidrules))

    # Second header row
    second_row = []
    for _ in datasets:
        second_row.append(metric_str_l)
        second_row.append(metric_str_r)

    lines.append(" & ".join([""] + second_row) + r" \\")
    lines.append(r"\midrule")

    # ========================
    # BASELINES
    # ========================

    for row in baseline_rows:
        lines.append(row + r" \\")

    if baseline_rows and method_rows:
        lines.append(r"\midrule")

    # ========================
    # METHODS
    # ========================

    for row in method_rows:
        lines.append(row + r" \\")

    # ========================
    # END
    # ========================

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        rf"\end{{{table_env}}}",
    ])

    return "\n".join(lines)

def build_baseline_rows(
    df,
    baseline_names,
    datasets,
    *,
    metric_l_col="recall@100",
    metric_r_col="ndcg@10",
    dec_pts=1,
    scale_100=True,
):
    """
    Builds LaTeX baseline rows in the same format as method_rows.

    df must be long format with columns:
        ['dataset', 'method', metric_l_col, metric_r_col]

    Returns: List[str]
    """

    rows = []

    for baseline in baseline_names:
        row_cells = [baseline]

        for ds in datasets:
            sub = df[(df["dataset"] == ds) & (df["method"] == baseline)]

            if sub.empty:
                row_cells.extend(["-", "-"])
                continue

            val_l = sub.iloc[0][metric_l_col]
            val_r = sub.iloc[0][metric_r_col]

            if scale_100:
                val_l *= 100
                val_r *= 100

            str_l = "-" if pd.isna(val_l) else f"{val_l:.{dec_pts}f}"
            str_r = "-" if pd.isna(val_r) else f"{val_r:.{dec_pts}f}"

            row_cells.extend([str_l, str_r])

        rows.append(" & ".join(row_cells))

    return rows

def get_mean_at_k(dir_path, metric, k):
    """
    Reads `all_queries_trec_eval_results.jsonl` from `dir_path`
    and returns the value of `mean_{metric}_{k}`.
    Example key: mean_P_1, mean_ndcg_10, etc.
    """
    file_path = os.path.join(dir_path, "all_queries_trec_eval_results.jsonl")

    if not os.path.exists(file_path):
        print(f"{file_path} does not exist.")
        return None

    # handle special case
    if metric in {'P', 'recall'}:
        key = f"mean_{metric}_{k}"
    else:
        key = f"mean_{metric}_cut_{k}"

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)  # treat as single JSON object

    if key not in data:
        print(f"{key} not found in {file_path}.")
        return None


    return data[key] * 100


def make_two_metric_table_q_dec(
    *,
    datasets: List[str],
    dataset_names: List[str],
    metric_str_l: str,
    metric_str_r: str,
    baseline_rows: List[str],
    method_rows: List[str],
    caption: str = "Q-Ref",
    label: str = "tab:q-ref",
    table_env: str = "table*",
    size_cmd: str = r"\small",
) -> str:
    """
    Same interface as make_two_metric_table.

    Assumes method_rows grouped in blocks of 4:
        [Org.-q, Basic, Q2D, EQR]

    Within each family:
        - Bold best per metric per dataset
        - Color bold:
            Org.-q -> black
            Basic  -> red
            Q2D    -> yellow
            EQR    -> green

    If ANY value is missing in a family:
        - No bolding
        - No coloring
        - No local summary
        - No contribution to global summary
    """

    assert len(method_rows) % 4 == 0, "method_rows must be grouped in 4s."

    def parse_row(row):
        parts = [p.strip() for p in row.split("&")]
        name = parts[0]
        vals = []
        for v in parts[1:]:
            v = v.strip()
            if v == "-" or v == "":
                vals.append(None)
            else:
                try:
                    vals.append(float(v))
                except:
                    vals.append(None)
        return name, vals

    color_map = {
        0: "black",   # Org.-q
        1: "red",     # Basic
        2: "yellow",  # Q2D
        3: "green",   # EQR
    }

    variant_name = {
        0: "Org.-q",
        1: "Basic",
        2: "Q2D",
        3: "EQR",
    }

    global_wins = {k: 0 for k in variant_name.values()}
    formatted_rows = []

    # ============================
    # PROCESS FAMILIES
    # ============================

    for fam_idx in range(0, len(method_rows), 4):

        block = method_rows[fam_idx:fam_idx+4]
        parsed = [parse_row(r) for r in block]

        names = [p[0] for p in parsed]
        values = [p[1] for p in parsed]

        num_metrics = len(values[0])

        # ---- Check for missing values ----
        family_has_missing = False

        for i in range(4):
            if len(values[i]) != num_metrics:
                family_has_missing = True
                break
            for v in values[i]:
                if v is None:
                    family_has_missing = True
                    break
            if family_has_missing:
                break

        # ----------------------------------
        # If missing → print raw only
        # ----------------------------------

        if family_has_missing:
            print(f"\nFamily skipped (missing values): {names[0].split(',')[0]}")

            for i in range(4):
                row_cells = [names[i]]
                for val in values[i]:
                    if val is None:
                        row_cells.append("-")
                    else:
                        row_cells.append(f"{val:.1f}")
                formatted_rows.append(" & ".join(row_cells))

            continue

        # ----------------------------------
        # Otherwise compute winners
        # ----------------------------------

        fam_wins = {k: 0 for k in variant_name.values()}
        best_indices = [None] * num_metrics

        for col in range(num_metrics):
            col_vals = [values[i][col] for i in range(4)]
            best_i = max(range(4), key=lambda i: col_vals[i])

            best_indices[col] = best_i
            fam_wins[variant_name[best_i]] += 1
            global_wins[variant_name[best_i]] += 1

        # Build formatted rows
        for i in range(4):
            row_cells = [names[i]]

            for col in range(num_metrics):
                val = values[i][col]
                cell = f"{val:.1f}"

                if best_indices[col] == i:
                    color = color_map[i]
                    cell = rf"\textbf{{\textcolor{{{color}}}{{{cell}}}}}"

                row_cells.append(cell)

            formatted_rows.append(" & ".join(row_cells))

        # Print local summary
        print(f"\nFamily: {names[0].split(',')[0]}")
        for k, v in fam_wins.items():
            print(f"  {k}: {v}")

    # ============================
    # GLOBAL SUMMARY
    # ============================

    print("\n===== GLOBAL WIN SUMMARY =====")
    for k, v in global_wins.items():
        print(f"{k}: {v}")

    if sum(global_wins.values()) > 0:
        overall_winner = max(global_wins.items(), key=lambda x: x[1])
        print(f"\nOverall Winner: {overall_winner[0]} ({overall_winner[1]} wins)")
    else:
        print("\nNo complete families available for global comparison.")

    # ============================
    # BUILD LATEX TABLE
    # ============================

    lines = []

    lines.extend([
        rf"\begin{{{table_env}}}[t]",
        size_cmd,
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
    ])

    col_spec = "l" + "cc" * len(datasets)
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    # Header
    top_row = [r"\multirow{2}{*}{Method}"]
    for ds_name in dataset_names:
        top_row.append(rf"\multicolumn{{2}}{{c}}{{{ds_name}}}")
    lines.append(" & ".join(top_row) + r" \\")

    cmidrules = []
    start_col = 2
    for i in range(len(datasets)):
        left = start_col + i * 2
        right = left + 1
        cmidrules.append(rf"\cmidrule(lr){{{left}-{right}}}")
    lines.append(" ".join(cmidrules))

    second_row = []
    for _ in datasets:
        second_row.append(metric_str_l)
        second_row.append(metric_str_r)
    lines.append(" & ".join([""] + second_row) + r" \\")
    lines.append(r"\midrule")

    # Baselines
    for row in baseline_rows:
        lines.append(row + r" \\")

    if baseline_rows and formatted_rows:
        lines.append(r"\midrule")

    # Methods
    for row in formatted_rows:
        lines.append(row + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        rf"\end{{{table_env}}}",
    ])

    return "\n".join(lines)