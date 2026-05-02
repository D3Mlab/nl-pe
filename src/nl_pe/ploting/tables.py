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


def make_std_two_metric_table(
    *,
    datasets: List[str],
    dataset_names: List[str],
    metric_str_l: str,
    metric_str_r: str,
    baseline_rows: List[str],
    method_rows: List[str],
    caption: str = "MAIN (STD)",
    label: str = "tab:main_std",
    table_env: str = "table*",
    size_cmd: str = r"\small",
) -> str:
    """
    Same interface as make_two_metric_table.

    This function intentionally keeps identical call semantics so existing notebook
    code can be reused by changing only the function name at the call-site.

    The caller should provide rows already populated with std values.
    """
    return make_two_metric_table(
        datasets=datasets,
        dataset_names=dataset_names,
        metric_str_l=metric_str_l,
        metric_str_r=metric_str_r,
        baseline_rows=baseline_rows,
        method_rows=method_rows,
        caption=caption,
        label=label,
        table_env=table_env,
        size_cmd=size_cmd,
    )


def make_ci_two_metric_table(
    *,
    datasets: List[str],
    dataset_names: List[str],
    metric_str_l: str,
    metric_str_r: str,
    baseline_rows: List[str],
    method_rows: List[str],
    caption: str = "MAIN (95% CI)",
    label: str = "tab:main_ci",
    table_env: str = "table*",
    size_cmd: str = r"\small",
) -> str:
    """
    Same interface as make_two_metric_table.

    Expects each numeric cell in baseline_rows/method_rows to already be formatted
    as "<upper>/<lower>" (or "-") by the caller.

    This keeps the call-site identical to make_two_metric_table except for the
    function name.
    """
    return make_two_metric_table(
        datasets=datasets,
        dataset_names=dataset_names,
        metric_str_l=metric_str_l,
        metric_str_r=metric_str_r,
        baseline_rows=baseline_rows,
        method_rows=method_rows,
        caption=caption,
        label=label,
        table_env=table_env,
        size_cmd=size_cmd,
    )


def _format_table_number(value, dec_pts=1):
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.{dec_pts}f}"


def _load_times_df(exp_dir):
    times_path = os.path.join(exp_dir, "times.csv")
    if not os.path.exists(times_path):
        return None

    try:
        return pd.read_csv(times_path)
    except Exception:
        return None


def _time_stat_from_df(df, value_col, stat, ignore_IO=False):
    if df is None or value_col not in df.columns:
        return None

    values = pd.to_numeric(df[value_col], errors="coerce")

    # Optional total-time correction removing IO contribution
    if ignore_IO and value_col == "tot":
        has_gp_inf = "gp_inf" in df.columns
        has_gp_inf_no_io = "gp_inf_no_IO" in df.columns

        if has_gp_inf and has_gp_inf_no_io:
            gp_inf = pd.to_numeric(df["gp_inf"], errors="coerce")
            gp_inf_no_io = pd.to_numeric(df["gp_inf_no_IO"], errors="coerce")
            io_vals = gp_inf - gp_inf_no_io
            values = values - io_vals
        elif has_gp_inf or has_gp_inf_no_io:
            # Ambiguous partial IO fields -> cannot safely recover IO-only component.
            values = pd.Series(np.nan, index=values.index)

    valid = values.dropna()
    if valid.empty:
        return None

    # Remove outliers using 1.5 IQR rule before computing mean/std
    q1 = valid.quantile(0.25)
    q3 = valid.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    valid = valid[(valid >= lower_bound) & (valid <= upper_bound)]
    if valid.empty:
        return None

    if stat == "mean":
        return float(valid.mean())
    if stat == "std":
        std_val = float(valid.std(ddof=1))
        return None if np.isnan(std_val) else std_val

    raise ValueError(f"Unsupported stat '{stat}'. Use 'mean' or 'std'.")


def _build_time_rows_from_specs(
    row_specs,
    datasets,
    *,
    stat,
    ignore_IO=False,
    dec_pts=1,
    trials_root=os.path.join("trials", "ir"),
):
    """
    row_specs supports either:
      - preformatted latex row string (passed through)
      - tuple/list: (display_name, relative_method_path)
        where final experiment path is trials_root/<dataset>/<relative_method_path>
    """
    rows = []

    for row_spec in row_specs:
        if isinstance(row_spec, str):
            rows.append(row_spec)
            continue

        if not (isinstance(row_spec, (list, tuple)) and len(row_spec) == 2):
            raise ValueError(
                "Each row spec must be either a preformatted string or "
                "(display_name, relative_method_path)."
            )

        method_name, rel_path = row_spec
        row_cells = [str(method_name)]

        for ds in datasets:
            exp_dir = os.path.join(trials_root, ds, str(rel_path))
            df = _load_times_df(exp_dir)

            llm_stat = _time_stat_from_df(
                df,
                "llm",
                stat,
                ignore_IO=False,
            )
            total_stat = _time_stat_from_df(
                df,
                "tot",
                stat,
                ignore_IO=ignore_IO,
            )

            row_cells.extend([
                _format_table_number(llm_stat, dec_pts=dec_pts),
                _format_table_number(total_stat, dec_pts=dec_pts),
            ])

        rows.append(" & ".join(row_cells))

    return rows


def make_two_metric_time_tables(
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
    ignore_IO: bool = False,
) -> str:
    """
    Temporal analysis twin-table builder.

    Accepts the same kwargs as make_two_metric_table, plus ignore_IO.
    It builds two tables using times.csv per experiment:
      1) mean table for [tot, llm]
      2) std table  for [tot, llm]

    If ignore_IO=True, the total metric uses:
        adjusted_tot = tot - (gp_inf - gp_inf_no_IO)
    whenever both gp_inf and gp_inf_no_IO exist.
    """

    mean_baselines = _build_time_rows_from_specs(
        baseline_rows,
        datasets,
        stat="mean",
        ignore_IO=ignore_IO,
    )
    mean_methods = _build_time_rows_from_specs(
        method_rows,
        datasets,
        stat="mean",
        ignore_IO=ignore_IO,
    )

    std_baselines = _build_time_rows_from_specs(
        baseline_rows,
        datasets,
        stat="std",
        ignore_IO=ignore_IO,
    )
    std_methods = _build_time_rows_from_specs(
        method_rows,
        datasets,
        stat="std",
        ignore_IO=ignore_IO,
    )

    mean_table = make_two_metric_table(
        datasets=datasets,
        dataset_names=dataset_names,
        metric_str_l=metric_str_l,
        metric_str_r=metric_str_r,
        baseline_rows=mean_baselines,
        method_rows=mean_methods,
        caption=f"{caption} (Mean)",
        label=f"{label}_mean",
        table_env=table_env,
        size_cmd=size_cmd,
    )

    std_table = make_two_metric_table(
        datasets=datasets,
        dataset_names=dataset_names,
        metric_str_l=metric_str_l,
        metric_str_r=metric_str_r,
        baseline_rows=std_baselines,
        method_rows=std_methods,
        caption=f"{caption} (STD)",
        label=f"{label}_std",
        table_env=table_env,
        size_cmd=size_cmd,
    )

    return mean_table + "\n\n\n" + std_table


def make_time_component_tables_per_dataset(
    *,
    datasets: List[str],
    dataset_names: List[str],
    baseline_rows: List[str],
    method_rows: List[str],
    caption: str = "Timing Component Breakdown",
    label: str = "tab:timing-components",
    table_env: str = "table*",
    size_cmd: str = r"\small",
    ignore_IO: bool = False,
    dec_pts: int = 1,
    trials_root: str = os.path.join("trials", "ir"),
) -> str:
    """
    Build one LaTeX table per dataset from times.csv files.

    Rows:
      - baseline_rows then method_rows
      - each row spec can be:
            1) preformatted latex row string
            2) (display_name, relative_method_path)

    Columns (each with mean/std):
      q-reform  <- q_gen
      knn       <- knn
      LLM Judge <- llm_obs
      GP        <- gp_inf (or gp_inf_no_IO when ignore_IO=True)
      MMR       <- mmr
    """
    if len(datasets) != len(dataset_names):
        raise ValueError("datasets and dataset_names must have the same length")

    gp_col = "gp_inf_no_IO" if ignore_IO else "gp_inf"
    component_defs = [
        ("q-reform", "q_gen"),
        ("knn", "knn"),
        ("LLM Judge", "llm_obs"),
        ("GP", gp_col),
        ("MMR", "mmr"),
    ]

    def build_rows_for_dataset(row_specs, dataset):
        rows = []

        for row_spec in row_specs:
            if isinstance(row_spec, str):
                rows.append(row_spec)
                continue

            if not (isinstance(row_spec, (list, tuple)) and len(row_spec) == 2):
                raise ValueError(
                    "Each row spec must be either a preformatted string or "
                    "(display_name, relative_method_path)."
                )

            method_name, rel_path = row_spec
            exp_dir = os.path.join(trials_root, dataset, str(rel_path))
            df = _load_times_df(exp_dir)

            row_cells = [str(method_name)]
            for _, src_col in component_defs:
                mean_val = _time_stat_from_df(df, src_col, stat="mean")
                std_val = _time_stat_from_df(df, src_col, stat="std")
                row_cells.extend([
                    _format_table_number(mean_val, dec_pts=dec_pts),
                    _format_table_number(std_val, dec_pts=dec_pts),
                ])

            rows.append(" & ".join(row_cells))

        return rows

    all_tables = []

    for dataset, dataset_name in zip(datasets, dataset_names):
        ds_baselines = build_rows_for_dataset(baseline_rows, dataset)
        ds_methods = build_rows_for_dataset(method_rows, dataset)

        lines = []
        ds_label_suffix = dataset.replace("-", "_")

        lines.extend([
            rf"\begin{{{table_env}}}[t]",
            size_cmd,
            r"\centering",
            rf"\caption{{{caption} ({dataset_name})}}",
            rf"\label{{{label}_{ds_label_suffix}}}",
        ])

        col_spec = "l" + "cc" * len(component_defs)
        lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
        lines.append(r"\toprule")

        top_row = [r"\multirow{2}{*}{Method}"]
        for display_name, _ in component_defs:
            top_row.append(rf"\multicolumn{{2}}{{c}}{{{display_name}}}")
        lines.append(" & ".join(top_row) + r" \\")

        cmidrules = []
        start_col = 2
        for i in range(len(component_defs)):
            left = start_col + i * 2
            right = left + 1
            cmidrules.append(rf"\cmidrule(lr){{{left}-{right}}}")
        lines.append(" ".join(cmidrules))

        second_row = ["", *[x for _ in component_defs for x in ("mean", "std")]]
        lines.append(" & ".join(second_row) + r" \\")
        lines.append(r"\midrule")

        for row in ds_baselines:
            lines.append(row + r" \\")

        if ds_baselines and ds_methods:
            lines.append(r"\midrule")

        for row in ds_methods:
            lines.append(row + r" \\")

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            rf"\end{{{table_env}}}",
        ])

        all_tables.append("\n".join(lines))

    return "\n\n\n".join(all_tables)

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


def get_std_at_k(dir_path, metric, k):
    """
    Reads `all_queries_trec_eval_results.jsonl` from `dir_path`
    and returns the value of `std_dev_<metric>_<k>`.
    Example key: std_dev_P_1, std_dev_ndcg_cut_10, etc.
    """
    file_path = os.path.join(dir_path, "all_queries_trec_eval_results.jsonl")

    if not os.path.exists(file_path):
        print(f"{file_path} does not exist.")
        return None

    if metric in {'P', 'recall'}:
        key = f"std_dev_{metric}_{k}"
    else:
        key = f"std_dev_{metric}_cut_{k}"

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if key not in data:
        print(f"{key} not found in {file_path}.")
        return None

    return data[key] * 100


def get_ci_at_k(dir_path, metric, k, dec_pts=1):
    """
    Returns CI text "<upper>/<lower>" using:
        mean ± 1.96 * std_dev
    from `all_queries_trec_eval_results.jsonl`.
    """
    mean_val = get_mean_at_k(dir_path, metric, k)
    std_val = get_std_at_k(dir_path, metric, k)

    if mean_val is None or std_val is None:
        return None

    upper = mean_val + 1.96 * std_val
    lower = mean_val - 1.96 * std_val
    return f"{upper:.{dec_pts}f}/{lower:.{dec_pts}f}"


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