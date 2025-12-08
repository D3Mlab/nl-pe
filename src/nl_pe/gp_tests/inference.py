import os
import pandas as pd
import sys
import json
import csv
from nl_pe.utils.setup_logging import setup_logging

class GPInference():
    def __init__(self, config):
        self.config = config
        self.logger = setup_logging(self.__class__.__name__, config = self.config, output_file=os.path.join(self.config['exp_dir'], "experiment.log"))
        self.logger.debug(f"Initializing {self.__class__.__name__} with config: {config}")

    def run_inference_test(self):

        #read from config
        n_obs = self.config.get('n_obs')
        n_unobs = self.config.get('n_unobs')
        d = self.config.get('d')

        #todo: device

        