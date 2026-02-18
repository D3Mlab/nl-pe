import json
import os
import time
import csv
from nl_pe.utils.setup_logging import setup_logging

class Scorer():

    def __init__(self, config):
        self.config = config
        self.logger = setup_logging(self.__class__.__name__, self.config)
        self.logger.debug(f"Initializing {self.__class__.__name__} with config: {config}")


class TextScorer(Scorer):
    def __init__(self, config):
        super().__init__(config)
        
        self.cache_path = self.config.get('data',{}).get('cache_path')
        self.texts_csv_path = self.config.get('data',{}).get('d_text_csv')

        # -------------------------
        # Load corpus texts ONCE
        # -------------------------
        self.logger.info("Loading document texts into memory...")
        start = time.time()

        self.doc_text_map = {}

        with open(self.texts_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.doc_text_map[str(row["d_id"])] = row["d_text"]

        elapsed = time.time() - start
        self.logger.info(
            f"Loaded {len(self.doc_text_map)} documents "
            f"in {elapsed:.3f}s"
        )

    def open_cache(self,qid,prompt_name=''):

        parts = [self.cache_path]
        if prompt_name:
            parts.append(prompt_name)
        parts.append(str(qid))

        dir_path = os.path.join(*parts)
        os.makedirs(dir_path, exist_ok=True)

        cache_path = os.path.join(dir_path, "cache.json")

        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                self.cache = json.load(f)
        else:
            self.cache = {}

        self.cache_file = cache_path  # store path for later writes

    def did_to_text(self,doc_ids, state):
        start = time.time()

        texts = []
        for did in doc_ids:
            did = str(did)
            if did not in self._doc_lookup:
                raise KeyError(f"Doc id {did} not found in corpus.")
            texts.append(self._doc_lookup[did])

        read_time = time.time() - start
        state["doc_text_read_times"].append(read_time)

        return texts