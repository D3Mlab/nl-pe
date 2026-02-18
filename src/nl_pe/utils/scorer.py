import json
import os
from nl_pe.utils.setup_logging import setup_logging

class Scorer():
    def __init__(self, config):
        self.config = config
        self.logger = setup_logging(self.__class__.__name__, self.config)
        
        self.cache_path = self.config.get('data',{}).get('cache_path')

    def open_cache(self,qid,prompt_name=''):
        # cache_path = os.path.join(self.cache_path,prompt_name,qid,'cache.json')
        # if os.path.exists(cache_path):
        #     with open(cache_path, "r", encoding="utf-8") as f:
        #         self.cache = json.load(f)
        # else:
        #     self.cache = {}

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