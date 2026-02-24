# clean_obs_times.py
#
# Usage:
#   python clean_obs_times.py "C:\path\to\experiments"
#
# What it does:
# - Recursively finds experiment directories (containing config.yaml)
# - Skips if no "ce" in experiment path
# - Reads k_acq and n_obs_iterations
# - Skips if k_acq <= 1
# - Recursively finds all detailed_results.json under the experiment dir
# - Greedily compresses observation_times
# - Prints one summary per experiment dir

import argparse
import json
import math
import shutil
from pathlib import Path
import yaml


def path_has_ce_dir(p: Path) -> bool:
    return any(part.lower() == "ce" for part in p.parts)


def load_k_acq_and_n_obs(config_path: Path):
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    al = (cfg or {}).get("active_learning", {}) or {}

    if "k_acq" not in al or "n_obs_iterations" not in al:
        return None

    k_acq = int(al["k_acq"])
    n_obs = int(al["n_obs_iterations"])

    if k_acq <= 1:
        return None

    return k_acq, n_obs


def greedy_compress(times: list[float], k_acq: int, n_obs_iterations: int) -> list[float]:
    target_len = math.ceil(n_obs_iterations / k_acq)
    t = list(times)

    while len(t) > target_len:
        if len(t) < k_acq:
            break

        window_sum = sum(t[:k_acq])
        best_sum = window_sum
        best_i = 0

        for i in range(1, len(t) - k_acq + 1):
            window_sum += t[i + k_acq - 1] - t[i - 1]
            if window_sum < best_sum:
                best_sum = window_sum
                best_i = i

        t = t[:best_i] + [float(best_sum)] + t[best_i + k_acq:]

    return t


def process_experiment_dir(exp_dir: Path) -> bool:
    config_path = exp_dir / "config.yaml"
    params = load_k_acq_and_n_obs(config_path)
    if params is None:
        return False

    k_acq, n_obs = params

    dr_files = list(exp_dir.rglob("detailed_results.json"))
    if not dr_files:
        return False

    changed_any = False

    for dr_path in dr_files:
        with dr_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        obs = data.get("observation_times")
        if not isinstance(obs, list):
            continue

        new_obs = greedy_compress(obs, k_acq, n_obs)

        if new_obs != obs:

            data["observation_times"] = new_obs
            with dr_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            changed_any = True

    status = "MOD" if changed_any else "OK"
    print(f"[{status}] {exp_dir} | k_acq={k_acq} n_obs={n_obs}")

    return changed_any


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=str)
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Root path does not exist: {root}")

    experiments_checked = 0
    experiments_modified = 0

    for config_path in root.rglob("config.yaml"):
        exp_dir = config_path.parent

        if not path_has_ce_dir(exp_dir):
            continue

        experiments_checked += 1

        if process_experiment_dir(exp_dir):
            experiments_modified += 1

    print(
        f"\nDone. experiments_checked={experiments_checked}, "
        f"modified={experiments_modified}"
    )


if __name__ == "__main__":
    main()