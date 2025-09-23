import os
import argparse
from nl_pe.eval_manager import EvalManager
from dotenv import load_dotenv

def run_eval_batch(e, skip_existing):
    # Find all config.yaml files in the directory and subdirectories
    exp_dirs = []
    for root, _, files in os.walk(e):
        if "eval_config.yaml" in files:
            exp_dirs.append(root)

    # Run evaluation for each experiment
    for exp_dir in exp_dirs:
        print(f"Evaluating experiment in directory: {exp_dir}")
        manager = EvalManager(exp_dir, skip_existing=skip_existing)
        manager.evaluate_experiment()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation for a batch of experiments in the specified directory.")
    parser.add_argument("--all", action="store_true", help="Run evaluation in all batch directories (True or False)")
    parser.add_argument("-e", type=str, help="Specific experiment batch path to evaluate")
    parser.add_argument("--skip_existing", action="store_true", help="Skip evaluation if output files already exist")
    args = parser.parse_args()

    datasets = ['dl19_bm25', 'trec_covid_bm25']
    models = ['4o/temp_1', 'sonnet/temp_1', 'nova/temp_1']
    base_dir = 'trials'

    if args.all:
        for dataset in datasets:
            for model in models:
                e = os.path.join(base_dir, dataset, model)
                if os.path.exists(e):
                    run_eval_batch(e, args.skip_existing)
                else:
                    print(f"Skipping missing directory: {e}")
    elif args.e:
        if os.path.exists(args.e):
            run_eval_batch(args.e, args.skip_existing)
        else:
            print(f"Specified experiment path does not exist: {args.e}")
    else:
        print("Please specify either --all True or specify an -e path.")
