import numpy as np
import pandas as pd
from nl_pe.utils.setup_logging import setup_logging
from pathlib import Path


class HyperpriorFitter():
    def __init__(self, config):
        self.config = config
        self.config['exp_dir'] = self.exp_dir
        self.logger = setup_logging(self.__class__.__name__, config = self.config, output_file=os.path.join(self.config['exp_dir'], "experiment.log"))          
     
    def fit_all(self):
        path = Path(self.exp_dir) / 'trained_params.csv'
        self.df = pd.read_csv(path)

        res = {}
        #if the csv has a column 'sig_noise'
    
    def fit_sig_noise(self,config,x,res):
          pass 
