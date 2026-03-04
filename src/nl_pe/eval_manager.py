import os
import yaml
import argparse
import pytrec_eval
import json
import csv
import numpy as np
from scipy.stats import norm
from nl_pe.utils.setup_logging import setup_logging
from nl_pe.utils.utils import get_doc_text_list
from nl_pe.utils.qrels import load_pytrec_eval_qrels
from pathlib import Path

VALID_DATASETS = {
    "nfcorpus",
    "robust04",
    "scifact",
    "trec-covid",
    "trec-news",
    "webis-touche2020",
}


class EvalManager:
    def __init__(self, eval_dir, skip_existing=False, do_times=False, skip_trec=False):
        self.eval_dir = eval_dir
        self.skip_existing = skip_existing
        self.do_times = do_times
        self.skip_trec = skip_trec
        self.load_config()
        self.setup_logger()

        self.selected_trec_measures = self.config.get("measures", pytrec_eval.supported_measures)
        self.qrels_path = self.config.get("qrels_path")
        self.qrels_dict = load_pytrec_eval_qrels(self.qrels_path)
        self.results_dir = Path(self.eval_dir) / "per_query_results"

        #check if all test queries in qrels are present in results_dir
        self.test_queries_path = Path(self.qrels_path).parent.parent / "test_queries.csv"
        self.check_test_query_coverage()

        # Load method lists from config
        self.per_query_methods = list(self.config.get("per_query_methods", []))
        self.all_query_methods = list(self.config.get("all_query_methods", []))
        self.required_files = list(self.config.get("required_files", []))

        #global containers here
        self.all_query_trec_eval_results = {}
        self.all_query_knn_times = {}
        self.all_query_times = {}
        self.time_columns = {"qid"}
        self.q_gen_times_cache = None

        self.eval_dir_path = Path(self.eval_dir)
        self.eval_path_parts = list(self.eval_dir_path.parts)
        self.dataset_name, self.exp_path_parts = self._extract_dataset_and_experiment_parts()
        self.primary_exp_type = self.exp_path_parts[0] if self.exp_path_parts else None

        self.configure_optional_evals()

    def evaluate_experiment(self):
        self.logger.info("Starting evaluation...")

        # Build required output file paths
        required_file_paths = [Path(self.eval_dir) / file for file in self.required_files]

        if self.skip_existing and all(file.exists() for file in required_file_paths):
            print(f"Skipping evaluation for {self.eval_dir} as all required output files already exist.")
            return

        if not self.results_dir.exists():
            print(f"Per query results directory {self.results_dir} does not exist")
            return None

        # Process per-query methods
        for query_dir in self.results_dir.iterdir():
            if query_dir.is_dir():
                self.curr_query_dir = query_dir
                self.curr_qid = query_dir.name
                self.curr_trec_file_path = Path(self.curr_query_dir) / "trec_results_raw.txt"
                self.curr_dedup_trec_file_path = Path(self.curr_query_dir) / "trec_results_deduplicated.txt"
                self.curr_query_detailed_results_path = Path(self.curr_query_dir) / "detailed_results.json"

                try:
                    # Run per-query methods specified in config
                    for method_name in self.per_query_methods:
                        if hasattr(self, method_name):
                            method = getattr(self, method_name)
                            #self.logger.debug(f"Running per-query method: {method_name} for query {self.curr_qid}")
                            method()
                        else:
                            self.logger.error(f"Method {method_name} not found in EvalManager class")
                except Exception as e:
                    self.logger.error(f"Error evaluating query {self.curr_qid}: {e}")

        # Run all-query methods specified in config
        for method_name in self.all_query_methods:
            if hasattr(self, method_name):
                method = getattr(self, method_name)
                self.logger.info(f"Running all-query method: {method_name}")
                method()
            else:
                self.logger.error(f"Method {method_name} not found in EvalManager class")

        self.logger.info("Evaluation completed.")

    def trec_evaluate_single_query(self):

        # Remove duplicates from TREC file and save deduplicated version
        deduped_lines = self.deduplicate_trec_results()

        # Parse deduplicated TREC results
        results = pytrec_eval.parse_run(deduped_lines)

        # Evaluate using pytrec_eval
        evaluator = pytrec_eval.RelevanceEvaluator(self.qrels_dict, self.selected_trec_measures)
        per_query_eval_results = evaluator.evaluate(results)

        # Store results for calculating means and std_devs
        self.all_query_trec_eval_results[self.curr_qid] = per_query_eval_results[self.curr_qid]

        # Write per-query trec evaluation results if not disabled
        if self.config.get('write_per_q_files'):
            curr_query_trec_eval_results_path = Path(self.curr_query_dir) / "trec_eval_results.jsonl"
            self.write_query_trec_jsonl(curr_query_trec_eval_results_path, per_query_eval_results)

    def deduplicate_trec_results(self):
        if not self.curr_trec_file_path.exists() or self.curr_trec_file_path.stat().st_size == 0:
            #if TREC run is empty or missing, write a dummy line
            qid = self.curr_trec_file_path.parent.name
            self.logger.warning(f"Query {qid}: TREC results file {self.curr_trec_file_path} is empty or missing. Adding a dummy line.")
            dummy_line = f"{qid} Q0 dummy_doc_id 1 0.0 dummy_run\n"
            with open(self.curr_dedup_trec_file_path, "w") as dedup_file:
                dedup_file.write(dummy_line)
            return [dummy_line]
        
        with open(self.curr_trec_file_path, "r") as trec_file:
            lines = trec_file.readlines()
            seen_docs = set()
            deduped_lines = []
            for line in lines:
                doc_id = line.split()[2]
                if doc_id not in seen_docs:
                    deduped_lines.append(line)
                    seen_docs.add(doc_id)
                else:
                    self.logger.warning(f"Query {self.curr_trec_file_path} has duplicate doc {doc_id}.")
        with open(self.curr_dedup_trec_file_path, "w") as dedup_file:
            dedup_file.writelines(deduped_lines)

        return deduped_lines

    def write_query_trec_jsonl(self, file_path, data):
        with open(file_path, "w") as file:
            for qid, eval_result in data.items():
                json.dump({"qid": qid, **eval_result}, file)
                file.write("\n")

    def write_all_queries_eval_trec_results(self):
        mean_results = {}
        std_dev_results = {}

        for measure in self.selected_trec_measures:
            values = [result.get(measure) for result in self.all_query_trec_eval_results.values() if result.get(measure) is not None]
            if values:
                mean_value = np.mean(values)
                std_dev = np.std(values, ddof=1) if len(values) > 1 else 0
                mean_results[f"mean_{measure}"] = mean_value
                std_dev_results[f"std_dev_{measure}"] = std_dev

        all_eval_results = {**mean_results, **std_dev_results}
        
        all_trec_eval_results_path = Path(self.eval_dir) / "all_queries_trec_eval_results.jsonl"

        with open(all_trec_eval_results_path, "w") as file:
            json.dump(all_eval_results, file)
            file.write("\n")

    def knn_time_single_query(self):
        #from self.curr_query_detailed_results_path , which is a json file,
        #get the value of "knn_time" and store it in self.all_query_knn_times
        try:
            with open(self.curr_query_detailed_results_path, 'r') as f:
                detailed_results = json.load(f)
                knn_time = detailed_results.get("knn_time")
                if knn_time is not None:
                    self.all_query_knn_times[self.curr_qid] = knn_time
                else:
                    pass
                    #self.logger.debug(f"Query {self.curr_qid}: knn_time not found in detailed_results.json")
        except FileNotFoundError:
            self.logger.error(f"Query {self.curr_qid}: detailed_results.json file not found at {self.curr_query_detailed_results_path}")
        except json.JSONDecodeError as e:
            self.logger.error(f"Query {self.curr_qid}: Error parsing JSON file: {e}")

    def write_all_queries_knn_times(self):
        #write a csv file all_queries_knn_times.csv with two columns: qid, knn_time based on self.all_query_knn_times
        if not self.all_query_knn_times:
        #    self.logger.warning("No KNN times found to write to CSV")
            return

        csv_path = Path(self.eval_dir) / "all_queries_knn_times.csv"

        try:
            with open(csv_path, 'w', newline='') as csvfile:
                fieldnames = ['qid', 'knn_time']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                # Write header
                writer.writeheader()

                # Write data rows
                for qid, knn_time in self.all_query_knn_times.items():
                    writer.writerow({'qid': qid, 'knn_time': knn_time})

            self.logger.info(f"Successfully wrote KNN times for {len(self.all_query_knn_times)} queries to {csv_path}")

        except Exception as e:
            self.logger.error(f"Error writing KNN times CSV file: {e}")

    def load_config(self):
        config_path = os.path.join(self.eval_dir, "eval_config.yaml")
        with open(config_path, "r") as config_file:
            self.config = yaml.safe_load(config_file)

    def setup_logger(self):
        self.logger = setup_logging(self.__class__.__name__, self.config, output_file=os.path.join(self.eval_dir, "evaluation.log"))

    def check_test_query_coverage(self):
        """
        Check that all test query ids from test_queries.csv exist as subdirectories
        in results_dir. Logs info if all present, warning for each missing query.
        """

        if not self.test_queries_path.exists():
            self.logger.warning(f"test_queries.csv not found at {self.test_queries_path} — skipping coverage check.")
            raise FileNotFoundError(f"test_queries.csv not found at {self.test_queries_path}")

        if not self.results_dir.exists():
            self.logger.warning(f"Results dir {self.results_dir} does not exist — skipping coverage check.")
            raise FileNotFoundError(f"Results dir {self.results_dir} does not exist")

        # ---- load expected qids from csv (first column, header-aware) ----
        expected_qids = []
        try:
            with open(self.test_queries_path, newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)  # skip header
                for row in reader:
                    if not row:
                        continue
                    expected_qids.append(str(row[0]).strip())
        except Exception as e:
            self.logger.error(f"Failed reading test_queries.csv at {self.test_queries_path}: {e}")
            return

        expected_qids_set = set(expected_qids)

        # ---- collect result qids from directory names ----
        result_qids_set = {
            p.name for p in self.results_dir.iterdir()
            if p.is_dir()
        }

        # ---- compare ----
        missing = sorted(expected_qids_set - result_qids_set)
        extra = sorted(result_qids_set - expected_qids_set)

        if not missing:
            self.logger.info(
                f"All {len(expected_qids_set)} test queries have result directories in {self.results_dir}."
            )
        else:
            for qid in missing:
                self.logger.warning(
                    f"Missing results directory for test query id: {qid}"
                )

        if extra:
            for qid in extra:
                self.logger.warning(
                    f"Results directory present but not in test_queries.csv: {qid}"
                )

    def _extract_dataset_and_experiment_parts(self):
        """
        Find the first path part that matches one of the known datasets.
        Everything after that is considered experiment path parts.
        """
        for i, part in enumerate(self.eval_path_parts):
            if part in VALID_DATASETS:
                dataset = part
                exp_parts = self.eval_path_parts[i + 1 :]
                return dataset, exp_parts

        return None, []

    def configure_optional_evals(self):
        """
        Adjust configured methods/required files based on CLI flags.
        """
        if self.do_times:
            if "time_single_query" not in self.per_query_methods:
                self.per_query_methods.append("time_single_query")
            if "write_all_queries_times" not in self.all_query_methods:
                self.all_query_methods.append("write_all_queries_times")
            if "times.csv" not in self.required_files:
                self.required_files.append("times.csv")

        if self.skip_trec:
            all_trec_eval_results_path = Path(self.eval_dir) / "all_queries_trec_eval_results.jsonl"
            if all_trec_eval_results_path.exists():
                self.per_query_methods = [
                    method_name
                    for method_name in self.per_query_methods
                    if method_name != "trec_evaluate_single_query"
                ]
                self.all_query_methods = [
                    method_name
                    for method_name in self.all_query_methods
                    if method_name != "write_all_queries_eval_trec_results"
                ]
                self.logger.info(
                    "-strec enabled and all_queries_trec_eval_results.jsonl exists. "
                    "Skipping TREC evaluation methods."
                )
            else:
                self.logger.info(
                    "-strec enabled but all_queries_trec_eval_results.jsonl does not exist. "
                    "Running TREC evaluation methods."
                )

    def _to_float_or_none(self, value, value_desc):
        if value is None:
            self.logger.warning(f"Query {self.curr_qid}: Missing value for {value_desc}.")
            return None

        try:
            return float(value)
        except (TypeError, ValueError):
            self.logger.warning(
                f"Query {self.curr_qid}: Could not parse {value_desc} value '{value}' as float."
            )
            return None

    def _get_float_from_detailed_results(self, detailed_results, key):
        if key not in detailed_results:
            self.logger.warning(f"Query {self.curr_qid}: Missing key '{key}' in detailed_results.json")
            return None
        return self._to_float_or_none(detailed_results.get(key), key)

    def _sum_list_from_detailed_results(self, detailed_results, key):
        if key not in detailed_results:
            self.logger.warning(f"Query {self.curr_qid}: Missing key '{key}' in detailed_results.json")
            return None

        values = detailed_results.get(key)
        if isinstance(values, list):
            total = 0.0
            for i, value in enumerate(values):
                parsed = self._to_float_or_none(value, f"{key}[{i}]")
                if parsed is None:
                    return None
                total += parsed
            return total

        self.logger.warning(
            f"Query {self.curr_qid}: Expected '{key}' to be a list in detailed_results.json, "
            f"got type {type(values).__name__}."
        )
        return self._to_float_or_none(values, key)

    def _sum_required_components(self, components, target_name):
        """
        Sum a list of tuples (component_name, component_value).
        Returns None if any component is None.
        """
        missing_components = [name for name, value in components if value is None]
        if missing_components:
            self.logger.warning(
                f"Query {self.curr_qid}: Could not compute '{target_name}' due to missing/invalid "
                f"components: {missing_components}"
            )
            return None

        return float(sum(value for _, value in components))

    def _load_q_gen_times(self):
        if self.q_gen_times_cache is not None:
            return

        self.q_gen_times_cache = {}

        if not self.dataset_name:
            self.logger.warning(
                "Could not determine dataset name from eval path; q_gen times will be unavailable."
            )
            return

        q_gen_times_csv_path = (
            Path("data")
            / "ir"
            / "beir"
            / self.dataset_name
            / "q_generation"
            / "gemini-2.5-flash-lite"
            / "5q"
            / "q_gen_times_and_parsing.csv"
        )

        if not q_gen_times_csv_path.exists():
            self.logger.warning(
                f"q_gen times file not found at {q_gen_times_csv_path}; q_gen values will be None."
            )
            return

        try:
            with open(q_gen_times_csv_path, "r", newline="", encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    qid = str(row.get("qid", "")).strip()
                    if not qid:
                        continue

                    gen_time = self._to_float_or_none(row.get("gen_time"), f"q_gen.gen_time[{qid}]")
                    if gen_time is not None:
                        self.q_gen_times_cache[qid] = gen_time
        except Exception as e:
            self.logger.warning(f"Failed to read q_gen times from {q_gen_times_csv_path}: {e}")

    def _get_q_gen_time_for_current_query(self):
        self._load_q_gen_times()

        if self.q_gen_times_cache is None:
            return None

        if self.curr_qid not in self.q_gen_times_cache:
            self.logger.warning(
                f"Query {self.curr_qid}: q_gen time not found in q_gen_times_and_parsing.csv"
            )
            return None

        return self.q_gen_times_cache[self.curr_qid]

    def _load_curr_query_detailed_results_for_timing(self):
        try:
            with open(self.curr_query_detailed_results_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.warning(
                f"Query {self.curr_qid}: detailed_results.json file not found at "
                f"{self.curr_query_detailed_results_path}"
            )
        except json.JSONDecodeError as e:
            self.logger.warning(
                f"Query {self.curr_qid}: Error parsing detailed_results.json: {e}"
            )
        except Exception as e:
            self.logger.warning(
                f"Query {self.curr_qid}: Unexpected error loading detailed_results.json: {e}"
            )

        return {}

    def time_single_query(self):
        row = {"qid": self.curr_qid}

        def set_col(col_name, value):
            row[col_name] = value
            self.time_columns.add(col_name)

        detailed_results = self._load_curr_query_detailed_results_for_timing()
        exp_type = self.primary_exp_type

        if exp_type == "dense":
            knn = self._get_float_from_detailed_results(detailed_results, "knn_time")
            set_col("knn", knn)
            set_col("tot", self._sum_required_components([("knn", knn)], "tot"))
            set_col("llm", 0.0)

        elif exp_type in {"dense_q_dec", "dense_q_dec_mmr"}:
            knn = self._get_float_from_detailed_results(detailed_results, "knn_time")
            q_gen = self._get_q_gen_time_for_current_query()
            set_col("knn", knn)
            set_col("q_gen", q_gen)
            set_col("tot", self._sum_required_components([("knn", knn), ("q_gen", q_gen)], "tot"))
            set_col("llm", q_gen)

        elif exp_type in {"dense_oracle", "lw", "dense_q_dec_oracle", "lw_q_dec"}:
            knn = self._get_float_from_detailed_results(detailed_results, "knn_time")
            llm_obs = self._sum_list_from_detailed_results(detailed_results, "observation_times")

            set_col("knn", knn)
            set_col("llm_obs", llm_obs)

            total_components = [("knn", knn), ("llm_obs", llm_obs)]
            llm_components = [("llm_obs", llm_obs)]

            if exp_type in {"dense_q_dec_oracle", "lw_q_dec"}:
                q_gen = self._get_q_gen_time_for_current_query()
                set_col("q_gen", q_gen)
                total_components.append(("q_gen", q_gen))
                llm_components.append(("q_gen", q_gen))

            set_col("tot", self._sum_required_components(total_components, "tot"))
            set_col("llm", self._sum_required_components(llm_components, "llm"))

        elif exp_type in {"gp_ws", "gp_ws_q_dec"}:
            knn = self._get_float_from_detailed_results(detailed_results, "knn_time")
            llm_obs = self._sum_list_from_detailed_results(detailed_results, "observation_times")
            final_inf_time = self._get_float_from_detailed_results(detailed_results, "final_inf_time")
            final_io_time = self._get_float_from_detailed_results(detailed_results, "final_IO_time")
            model_update_times = self._sum_list_from_detailed_results(detailed_results, "model_update_times")

            gp_inf = self._sum_required_components(
                [
                    ("final_inf_time", final_inf_time),
                    ("final_IO_time", final_io_time),
                    ("model_update_times", model_update_times),
                ],
                "gp_inf",
            )
            gp_inf_no_io = self._sum_required_components(
                [
                    ("final_inf_time", final_inf_time),
                    ("model_update_times", model_update_times),
                ],
                "gp_inf_no_IO",
            )

            set_col("knn", knn)
            set_col("llm_obs", llm_obs)
            set_col("gp_inf", gp_inf)
            set_col("gp_inf_no_IO", gp_inf_no_io)

            llm_components = [("llm_obs", llm_obs)]
            tot_components = [("knn", knn), ("llm_obs", llm_obs), ("gp_inf", gp_inf)]

            if exp_type == "gp_ws_q_dec":
                q_gen = self._get_q_gen_time_for_current_query()
                set_col("q_gen", q_gen)
                llm_components.append(("q_gen", q_gen))
                tot_components.append(("q_gen", q_gen))

            set_col("llm", self._sum_required_components(llm_components, "llm"))
            set_col("tot", self._sum_required_components(tot_components, "tot"))

        elif exp_type in {"gp_al", "gp_al_q_dec"}:
            llm_obs = self._sum_list_from_detailed_results(detailed_results, "observation_times")
            final_inf_time = self._get_float_from_detailed_results(detailed_results, "final_inf_time")
            final_io_time = self._get_float_from_detailed_results(detailed_results, "final_IO_time")
            model_update_times = self._sum_list_from_detailed_results(detailed_results, "model_update_times")
            inner_acquisition_times = self._sum_list_from_detailed_results(detailed_results, "inner_acquisition_times")
            inner_acquisition_io_times = self._sum_list_from_detailed_results(detailed_results, "inner_acquisition_IO_times")
            inner_acquisition_sort_times = self._sum_list_from_detailed_results(detailed_results, "inner_acquisition_sort_times")

            gp_inf = self._sum_required_components(
                [
                    ("final_inf_time", final_inf_time),
                    ("final_IO_time", final_io_time),
                    ("model_update_times", model_update_times),
                    ("inner_acquisition_times", inner_acquisition_times),
                    ("inner_acquisition_IO_times", inner_acquisition_io_times),
                    ("inner_acquisition_sort_times", inner_acquisition_sort_times),
                ],
                "gp_inf",
            )
            gp_inf_no_io = self._sum_required_components(
                [
                    ("final_inf_time", final_inf_time),
                    ("model_update_times", model_update_times),
                    ("inner_acquisition_times", inner_acquisition_times),
                    ("inner_acquisition_sort_times", inner_acquisition_sort_times),
                ],
                "gp_inf_no_IO",
            )

            set_col("llm_obs", llm_obs)
            set_col("gp_inf", gp_inf)
            set_col("gp_inf_no_IO", gp_inf_no_io)

            llm_components = [("llm_obs", llm_obs)]
            tot_components = [("gp_inf", gp_inf), ("llm_obs", llm_obs)]

            if "mmr_af" in self.exp_path_parts:
                mmr = self._sum_list_from_detailed_results(detailed_results, "mmr_knn_times")
                set_col("mmr", mmr)
                tot_components.append(("mmr", mmr))

            if exp_type == "gp_al_q_dec":
                q_gen = self._get_q_gen_time_for_current_query()
                set_col("q_gen", q_gen)
                llm_components.append(("q_gen", q_gen))
                tot_components.append(("q_gen", q_gen))

            set_col("llm", self._sum_required_components(llm_components, "llm"))

            set_col("tot", self._sum_required_components(tot_components, "tot"))

        else:
            self.logger.warning(
                f"Query {self.curr_qid}: Unsupported experiment type for timing analysis: "
                f"'{exp_type}'. Writing None for tot/llm."
            )
            set_col("tot", None)
            set_col("llm", None)

        self.all_query_times[self.curr_qid] = row

    def write_all_queries_times(self):
        if not self.all_query_times:
            self.logger.warning("No per-query timing rows found to write to times.csv")
            return

        csv_path = Path(self.eval_dir) / "times.csv"

        preferred_order = [
            "qid",
            "q_gen",
            "knn",
            "llm_obs",
            "gp_inf",
            "gp_inf_no_IO",
            "mmr",
            "tot",
            "llm",
        ]

        ordered_columns = [col for col in preferred_order if col in self.time_columns]
        extra_columns = sorted(col for col in self.time_columns if col not in preferred_order)
        fieldnames = ordered_columns + extra_columns

        try:
            with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for qid in sorted(self.all_query_times.keys()):
                    row = self.all_query_times[qid]
                    writer.writerow({field: row.get(field) for field in fieldnames})

            self.logger.info(
                f"Successfully wrote timing analysis for {len(self.all_query_times)} queries to {csv_path}"
            )
        except Exception as e:
            self.logger.error(f"Error writing times.csv: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate an experiment based on a config file.")
    parser.add_argument("-c", "--eval-dir", type=str, help="The path to the evaluation dir containing eval_config.yaml")
    parser.add_argument("-se", "--skip-existing", action="store_true", help="Skip evaluation if output files already exist.")
    parser.add_argument("-t", "--times", action="store_true", help="Do timing eval or not")
    parser.add_argument(
        "-strec",
        "--skip-trec",
        action="store_true",
        help=(
            "Skip TREC evaluation methods when all_queries_trec_eval_results.jsonl "
            "already exists in the eval directory."
        ),
    )
    args = parser.parse_args()

    eval_manager = EvalManager(
        args.eval_dir,
        skip_existing=args.skip_existing,
        do_times=args.times,
        skip_trec=args.skip_trec,
    )
    eval_manager.evaluate_experiment()
