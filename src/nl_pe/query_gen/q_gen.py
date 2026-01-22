import pandas as pd
import os
import csv
from pathlib import Path
import json
from nl_pe.utils.setup_logging import setup_logging
from nl_pe.llm.prompter import Prompter

class QueryGenerator():
    def __init__(self,config):
        self.config = config
        self.exp_dir = self.config['exp_dir']
        self.logger = setup_logging(self.__class__.__name__, config = self.config, output_file=os.path.join(self.config['exp_dir'], "experiment.log"))       

    def generate(self):
        self.logger.info(f"Starting query generation in {self.exp_dir}")
        
        # ---------------------
        # read config params
        # ---------------------

        # query generation params
        self.k_new_qs = self.config.get("query_gen", {}).get("k_new_qs")

        # I/O
        path_to_test_qs = self.config.get("data", {}).get("queries_csv_path")
        f_make_writer = getattr(self, self.config.get("data", {}).get("f_make_writer"))
        

        # prompt / template params
        template_path = self.config.get("templates", {}).get("template_path")
        f_make_prompt_dict = getattr(self, self.config['templates'].get('f_make_prompt_dict'))
        f_write_row = getattr(self, self.config['templates'].get('f_write_row'))

        # debug logging
        self.logger.info(f"k_new_qs: {self.k_new_qs}")
        self.logger.info(f"queries_csv_path: {path_to_test_qs}")
        self.logger.info(f"Template: {template_path}")

        #------------------
        #Setup
        #-------------------
        # read in the queries to process as a list of strings, q_texts
        df = pd.read_csv(path_to_test_qs)
        q_ids = df["q_id"].tolist()
        q_texts = df["q_text"].tolist()

        #Setup output csv
        self.output_path = Path(self.exp_dir) / f"gen_qs.csv" 
        self.writer = f_make_writer()

        #Prompter
        prompter = Prompter(self.config)


        for q_id, q in zip(q_ids, q_texts):

            prompt_dict = f_make_prompt_dict(q)
            response = prompter.prompt_from_temp(template_path, prompt_dict)
            self.logger.debug(f"Full response for q_id={q_id}:", response)

            f_write_row(q_id, q, response)

    def _write_q_decomp_row(self, q_id, q, response):
        
        row = {
            "q_id": q_id,
            "q_0": q,
        }

        new_queries = None

        # note that the prompter adds "JSON_dict" to the response if it can parse it
        if isinstance(response, dict) and "JSON_dict" in response:
            json_data = response["JSON_dict"]

            # extract q_1, ..., q_k_new_qs from json_data, which has the format:
            #
            # <JSON format template>
            # {new_query_list: [<new_query_1>,...,<new_query_{{K}}>]
            # </JSON format template>
            if isinstance(json_data, dict):
                new_queries = json_data.get("new_query_list")

        # if anything went wrong or the structure isn't as expected, handle below
        if not isinstance(new_queries, list):
            new_queries = []

        # fill q_1 ... q_k_new_qs
        for i in range(1, self.k_new_qs + 1):
            if i <= len(new_queries):
                row[f"q_{i}"] = new_queries[i - 1]
            else:
                # qi, ..., q_k_new_qs should be recorded as PARSING_ERROR
                row[f"q_{i}"] = "PARSING_ERROR"

        self.writer.writerow(row)    
    
    def _make_q_decomp_prompt_dict(self, q):
        prompt_dict = {
            "q": q,
            "k_new": self.k_new_qs,
            "k_tot": self.k_new_qs + 1
        }
        return prompt_dict
    
    def _make_q_decomp_writer(self):
        # this new .csv will have with columns:
        # 'q_id', 'q_0', 'q_1', ..., 'q_{k_new_qs}'
        # where q_0 is the original query, and q_1 ... q_k_new_qs are the new decomposed queries
        fieldnames = ["q_id"] + [f"q_{i}" for i in range(0, self.k_new_qs + 1)]
        self._csv_file = open(self.output_path, "w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self._csv_file, fieldnames=fieldnames)
        self.writer.writeheader()
        return self.writer
    
    def _make_eqr_writer(self):
        # this new .csv will have columns:
        # 'q_id',
        # 'q_0', 'q_1', ..., 'q_{k_new_qs}',
        # 'k_0', 'k_1', ..., 'k_{k_new_qs}'

        fieldnames = (
            ["q_id"]
            + [f"q_{i}" for i in range(0, self.k_new_qs + 1)]
            + [f"k_{i}" for i in range(0, self.k_new_qs + 1)]
        )

        self._csv_file = open(self.output_path, "w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self._csv_file, fieldnames=fieldnames)
        self.writer.writeheader()
        return self.writer
