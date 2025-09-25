import json
import re
import jinja2
from nl_pe.utils.setup_logging import setup_logging
from nl_pe.utils.utils import *
from nl_pe.llm import LLM_CLASSES
import argparse
import yaml
from dotenv import load_dotenv

class Prompter():
    
    def __init__(self,config):

        self.config = config
        self.logger = setup_logging(self.__class__.__name__, self.config)

        self.llm_config = config.get('llm', {}) 
        self.model_class_name = self.llm_config.get('model_class')
        self.model_name = self.llm_config.get('model_name')
        self.template_config = config.get("templates", {})
        self.template_dir = self.template_config.get('template_dir')

        model_class = LLM_CLASSES.get(self.model_class_name)
        self.llm = model_class(config,self.model_name)
        self.jinja_env = jinja2.Environment(loader=jinja2.FileSystemLoader(searchpath=self.template_dir))


    def pw_rerank(self, state):

        if state["current_batch"] == None:
            return

        instance = state['instance']
        query = instance["query"]["text"]

        psg_list_batch = state["current_batch"]

        # Get local passages for the batch using simple pIDs (e.g. 'p1' instead of '1k43hj2f53l345')
        #local_psgs = {<local_p_id>: {p_id: __, text: __},...}
        local_psgs = self.get_local_psgs(psg_list_batch)

        #get label descriptions (e.g. "3 is highly relevant...", etc)
        label_macro_name = self.config['templates'].get('label_macro_name')
        n_labels = self.config['templates'].get('n_labels')
        list_len = len(local_psgs)

        prompt_dict = {
            'query' : query,
            'local_p_ids' : list(local_psgs.keys()),
            'p_texts' : [psg["text"] for psg in psg_list_batch],
            'label_macro_name': label_macro_name,
            'n_labels': n_labels,
            'list_len': list_len
        }

        template_path = self.template_config["pw_rerank"]
        prompt = self.render_prompt(prompt_dict, template_path)

        llm_response = self.llm.prompt(prompt)
        llm_output = llm_response["message"]
        self.add_response_to_state(state,llm_output)
        self.add_prompt_tokens_to_state(state,llm_response)

        scores = self.parse_llm_list_pw(llm_output)
        scores = [int(score) for score in scores]

        if 'batch_scores' not in state:
            state['batch_scores'] = []
        state['batch_scores'].append(scores)

        self.logger.debug(f"pw_rerank scores: {scores}")

        # Ensure pid_to_score_dict exists in state
        if "pid_to_score_dict" not in state:
            state["pid_to_score_dict"] = {}

        for pid in [psg["pid"] for psg in psg_list_batch]:
            if pid not in state["pid_to_score_dict"]:
                state["pid_to_score_dict"][pid] = []    

        # Extend the scores for the pids in the batch
        for pid, score in zip([psg["pid"] for psg in psg_list_batch], scores):
            state["pid_to_score_dict"][pid].append(score)

        duration = llm_response["prompt_time"]
        self.logger.debug(f"pw_rerank duration: {duration}")

        if 'prompting_runtimes' not in state:
            state['prompting_runtimes'] = []
        state['prompting_runtimes'].append(duration)

    def add_prompt_to_state(self,state,prompt):
        if "prompts" not in state:
            state["prompts"] = []  
        state["prompts"].append(prompt)

    def add_response_to_state(self,state,response):
        if "responses" not in state:
            state["responses"] = []  
        state["responses"].append(response)

    def add_prompt_tokens_to_state(self,state,llm_response):
        if llm_response.get('prompt_tokens'):
            if "prompt_tokens" not in state:
                state["prompt_tokens"] = []  
            state["prompt_tokens"].append(llm_response['prompt_tokens'])

    def prompt_from_temp(self,template_path, prompt_dict = {}):
        prompt = self.render_prompt(prompt_dict, template_path)
        return self.prompt_from_str(prompt)

    def prompt_from_str(self, prompt):
        """
        Generic method to call the LLM with a prompt and return the full response.

        Args:
            prompt (str): The prompt string to send to the LLM

        Returns:
            dict: The full response from the LLM including message, timing, and token counts
        """
        return self.llm.prompt(prompt)

    def render_prompt(self, prompt_dict, template_path):
        template = self.jinja_env.get_template(template_path)
        return template.render(prompt_dict)


  

if __name__ == "__main__":
    #temporary testing for prompter
    parser = argparse.ArgumentParser(description="Test the Prompter class.")
    parser.add_argument("-c", "--config_path", type=str, help="The path to the config file.")
    args = parser.parse_args()

    with open(args.config_path, "r") as config_file:
        config = yaml.safe_load(config_file)

    load_dotenv()

    prompter = Prompter(config)
