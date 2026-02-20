#!/usr/bin/env python3

import sys
import os
import pandas as pd


def parse_int_list(s: str):
    try:
        return sorted(set(int(x.strip()) for x in s.split(",")))
    except ValueError:
        raise ValueError("Second argument must be a comma-separated list of integers (e.g., '5,10').")


def get_max_q_index(columns):
    """
    Extract the maximum q_i index from columns like q_0, q_1, ...
    """
    q_indices = []
    for col in columns:
        if col.startswith("q_"):
            try:
                q_indices.append(int(col.split("_")[1]))
            except ValueError:
                continue
    return max(q_indices) if q_indices else -1


def main():
    if len(sys.argv) != 3:
        print("Usage: python reduce_q_cnt.py <csv_path> <comma_separated_ints>")
        sys.exit(1)

    csv_path = sys.argv[1]
    int_list_str = sys.argv[2]

    if not os.path.isfile(csv_path):
        print(f"File not found: {csv_path}")
        sys.exit(1)

    try:
        target_counts = parse_int_list(int_list_str)
    except ValueError as e:
        print(str(e))
        sys.exit(1)

    if not target_counts:
        print("No valid integers provided.")
        sys.exit(1)

    df = pd.read_csv(csv_path)

    if "q_id" not in df.columns:
        print("CSV must contain 'q_id' column.")
        sys.exit(1)

    max_existing_q = get_max_q_index(df.columns)
    max_requested_q = max(target_counts)

    # Assert CSV contains enough q_i columns
    if max_existing_q < max_requested_q - 1:
        print(
            f"Warning: CSV only contains up to q_{max_existing_q}, "
            f"but requested up to q_{max_requested_q - 1}. Doing nothing."
        )
        sys.exit(0)

    # Process each requested truncation
    for i in target_counts:
        out_dir = f"{i}q"
        os.makedirs(out_dir, exist_ok=True)

        # Keep q_id and q_0 ... q_{i-1}
        cols_to_keep = ["q_id"] + [f"q_{j}" for j in range(i)]

        truncated_df = df[cols_to_keep]

        out_path = os.path.join(out_dir, "gen_qs.csv")
        truncated_df.to_csv(out_path, index=False)

        print(f"Created: {out_path}")


if __name__ == "__main__":
    main()