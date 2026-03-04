import os
import argparse
import subprocess
import sys
from dotenv import load_dotenv
from pathlib import Path


def _get_validation_failure_reason(exp_dir, valid_l1, valid_l2, not_valid_parts):
    """
    Returns None if exp_dir is valid; otherwise returns a string reason.

    Rules:
      1) Must contain one of valid_l1 path parts (dataset).
      2) The part immediately after dataset must be in valid_l2.
      3) If any part is in not_valid_parts, skip.
    """
    exp_parts = list(Path(exp_dir).parts)

    # Rule 3: reject explicit banned parts anywhere in the path
    banned_hit = next((part for part in exp_parts if part in not_valid_parts), None)
    if banned_hit is not None:
        return f"contains banned part '{banned_hit}'"

    # Rule 1: find dataset part
    dataset_idx = None
    for i, part in enumerate(exp_parts):
        if part in valid_l1:
            dataset_idx = i
            break

    if dataset_idx is None:
        return f"no dataset part found in valid_l1={sorted(valid_l1)}"

    # Rule 2: validate method right after dataset
    if dataset_idx + 1 >= len(exp_parts):
        return "missing method part after dataset"

    l2 = exp_parts[dataset_idx + 1]
    if l2 not in valid_l2:
        return f"invalid l2 method '{l2}' (expected one of {sorted(valid_l2)})"

    return None


def run_eval_batch(e, skip_existing, times=False, skip_trec=False, valid_only=False):
    # Load environment variables
    load_dotenv()

    # Find all eval_config.yaml files in the directory and subdirectories
    exp_dirs = []
    for root, _, files in os.walk(e):
        if "eval_config.yaml" in files:
            exp_dirs.append(root)

    # Hardcoded validation rules used only when -valid is enabled
    valid_l1 = {
        "nfcorpus",
        "robust04",
        "scifact",
        "trec-covid",
        "trec-news",
        "webis-touche2020"
    }
    valid_l2 = {
        "dense",
        "dense_oracle",
        "dense_q_dec_mmr",
        "gp_ws",
        "gp_al",
        "gp_al_q_dec",
        "gp_ws_q_dec",
        "dense_q_dec",
        "dense_q_dec_oracle",
        "lw",
        "lw_q_dec",
    }
    # Add path parts here if you want to force-skip matching experiments.
    not_valid_parts = {'gt','gpt-5.2','gpt-5-nano'}

    # Run evaluation for each experiment using subprocess with new cmd line interface
    for exp_dir in exp_dirs:
        if valid_only:
            invalid_reason = _get_validation_failure_reason(
                exp_dir,
                valid_l1=valid_l1,
                valid_l2=valid_l2,
                not_valid_parts=not_valid_parts,
            )
            if invalid_reason is not None:
                print(f"Skipping experiment in directory: {exp_dir} ({invalid_reason})")
                continue

        print(f"Evaluating experiment in directory: {exp_dir}")
        cmd = [sys.executable, 'src/nl_pe/eval_manager.py', '-c', exp_dir]
        if skip_existing:
            cmd.append('--skip-existing')
        if times:
            cmd.append('--times')
        if skip_trec:
            cmd.append('--skip-trec')
        subprocess.run(cmd, cwd='.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation for a batch of experiments in the specified directory.")
    parser.add_argument("-c", "--eval-dir", type=str, required=True, help="The path to the directory containing the batch of evaluations.")
    parser.add_argument("-se","--skip-existing", action="store_true", help="Skip evaluation if output files already exist")
    parser.add_argument("-t", "--times", action="store_true", help="Run timing analysis in eval manager")
    parser.add_argument(
        "-strec",
        "--skip-trec",
        action="store_true",
        help="Skip TREC eval methods when all_queries_trec_eval_results.jsonl already exists",
    )
    parser.add_argument(
        "-v",
        "--valid-only",
        action="store_true",
        help="Only run experiments matching hardcoded valid path rules in this script",
    )
    args = parser.parse_args()

    run_eval_batch(
        args.eval_dir,
        args.skip_existing,
        times=args.times,
        skip_trec=args.skip_trec,
        valid_only=args.valid_only,
    )
