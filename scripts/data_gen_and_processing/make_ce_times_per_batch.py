# clean_obs_times.py
#
# Usage:
#   python clean_obs_times.py "C:\path\to\experiments"
#
# What it does:
# - Recursively finds any ".../detailed_results.json" whose PATH contains a directory
#   with "ce" in its name (case-insensitive).
# - For each hit:
#   - reads config.yaml from (detailed_results.json).parent.parent / "config.yaml"
#   - extracts active_learning.k_acq and active_learning.n_obs_iterations
#   - greedily compresses observation_times until length <= ceil(n_obs_iterations / k_acq):
#       repeatedly replace the contiguous window of length k_acq with the smallest sum
#       by that sum (single element).
#   - writes the modified JSON back in-place (with a .bak backup)

import argparse
import json
import math
import shutil
from pathlib import Path

import yaml


def path_has_ce_dir(p: Path) -> bool:
    # "dir 'ce' somewhere in the name" => any path component contains "ce"
    # e.g., ".../dense_oracle/ce/l6/..." or ".../cache_ce_l6/..."
    return any("ce" in part.lower() for part in p.parts)


def load_k_acq_and_n_obs(config_path: Path):
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    al = (cfg or {}).get("active_learning", {}) or {}

    if "k_acq" not in al or "n_obs_iterations" not in al:
        return None

    k_acq = int(al["k_acq"])
    n_obs = int(al.get("n_obs_iterations"))

    if k_acq <= 0:
        return None

    return k_acq, n_obs


def greedy_compress(times: list[float], k_acq: int, n_obs_iterations: int) -> list[float]:
    # target number of batch-timing entries
    target_len = math.ceil(n_obs_iterations / k_acq) if k_acq > 0 else len(times)
    if target_len < 0:
        target_len = 0

    # Work on a copy
    t = list(times)

    # If k_acq==1, no compression possible/needed beyond target_len trimming logic
    if k_acq <= 1:
        return t[:target_len] if len(t) > target_len else t

    # Greedily compress until we reach target_len
    while len(t) > target_len:
        if len(t) < k_acq:
            # can't take a full window; stop
            break

        # Find window of length k_acq with smallest sum
        # Sliding window O(n)
        window_sum = sum(t[:k_acq])
        best_sum = window_sum
        best_i = 0

        for i in range(1, len(t) - k_acq + 1):
            window_sum += t[i + k_acq - 1] - t[i - 1]
            if window_sum < best_sum:
                best_sum = window_sum
                best_i = i

        # Replace that window with its sum (single element)
        t = t[:best_i] + [float(best_sum)] + t[best_i + k_acq :]

    return t


def process_detailed_results(dr_path: Path, make_backup: bool = True) -> bool:
    # returns True if modified
    config_path = dr_path.parent.parent.parent / "config.yaml"
    if not config_path.exists():
        print(f"[SKIP] No config.yaml at expected location: {config_path}")
        return False

    params = load_k_acq_and_n_obs(config_path)
    if params is None:
        print(f"[SKIP] No k_acq in config: {config_path}")
        return False

    k_acq, n_obs = params

    with dr_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "observation_times" not in data or not isinstance(data["observation_times"], list):
        print(f"[SKIP] No observation_times list in: {dr_path}")
        return False

    old = data["observation_times"]
    new = greedy_compress(old, k_acq=k_acq, n_obs_iterations=n_obs)

    if new == old:
        print(f"[OK]   No change: {dr_path} (k_acq={k_acq}, n_obs_iterations={n_obs}, len={len(old)})")
        return False

    if make_backup:
        bak = dr_path.with_suffix(dr_path.suffix + ".bak")
        if not bak.exists():
            shutil.copy2(dr_path, bak)

    data["observation_times"] = new
    with dr_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(
        f"[MOD]  {dr_path}\n"
        f"       k_acq={k_acq}, n_obs_iterations={n_obs}, "
        f"len {len(old)} -> {len(new)}"
    )
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=str, help="Root directory to search")
    ap.add_argument("--no-backup", action="store_true", help="Do not write .bak backups")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Root path does not exist: {root}")

    hits = 0
    mods = 0

    for dr in root.rglob("detailed_results.json"):
        if not path_has_ce_dir(dr):
            continue
        hits += 1
        if process_detailed_results(dr, make_backup=False):
            mods += 1

    print(f"\nDone. hits={hits}, modified={mods}")


if __name__ == "__main__":
    main()