import json
import re
import jinja2
from llm_passage_ranking.utils.setup_logging import setup_logging
from llm_passage_ranking.utils.utils import *
from llm_passage_ranking.llm import LLM_CLASSES
import argparse
import yaml
import os
from dotenv import load_dotenv
import time

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

    def lw_rerank(self, state):

        if state["current_batch"] == None:
            return

        instance = state['instance']
        query = instance["query"]["text"]
        psg_list_batch = state["current_batch"]

        #if config['data']['use_local_p_ids'], get simplified p_ids (e.g. 'p1' instead of '1k43hj2f53l345')
        #local_psgs = {<local_p_id>: {p_id: __, text: __},...}
        local_psgs = self.get_local_psgs(psg_list_batch)

        k = len(local_psgs)

        prompt_dict = {
            'query' : query,
            'local_p_ids' : list(local_psgs.keys()),
            'p_texts' : [psg["text"] for psg in instance["psg_list"]],
            'k': k
        }

        template_dir = self.template_config["lw_rerank"]
        prompt = self.render_prompt(prompt_dict, template_dir)

        llm_response = self.llm.prompt(prompt)
        llm_output = llm_response["message"]
        self.add_response_to_state(state,llm_output)
        self.add_prompt_tokens_to_state(state,llm_response)
        
        #list of strings (pids) that can contain duplicates and non-existent pids 
        raw_batch_pids = self.parse_llm_list_lw(llm_output)

        #if pad = False, output list may be shorter than batch size
        pad = self.config['rerank'].get('lw_padding', False)

        self.update_lw_pid_outputs(state,raw_batch_pids, local_psgs, pad)

        duration = llm_response["prompt_time"]
        self.logger.debug(f"lw_rerank duration: {duration}")
        if 'prompting_runtimes' not in state:
            state['prompting_runtimes'] = []
        state['prompting_runtimes'].append(duration)


    def update_lw_pid_outputs(self,state,raw_batch_pids, local_psgs, pad):
        # processes parsed pids for hallucinations and duplicates and writes results to state
        self.logger.debug(f"lw_rerank raw_batch_pids: {raw_batch_pids}")
        if 'raw_batch_local_pid_lists' not in state:
            state['raw_batch_local_pid_lists'] = []
        state['raw_batch_local_pid_lists'].append(raw_batch_pids)

        seen_pids = set()
        valid_pid_list = []
        duplicated_pids = []
        nonexistent_pids = []
        for raw_pid in raw_batch_pids:
            if raw_pid in local_psgs:
                pid = local_psgs[raw_pid]['pid']
                if pid not in seen_pids:
                    seen_pids.add(pid)
                    valid_pid_list.append(pid)
                else:
                    duplicated_pids.append(pid)
            else:
                nonexistent_pids.append(raw_pid)

        if 'duplicated_batch_pid_lists' not in state:
            state['duplicated_batch_pid_lists'] = []
        state['duplicated_batch_pid_lists'].append(duplicated_pids)

        if 'nonexistent_batch_pid_lists' not in state:
            state['nonexistent_batch_pid_lists'] = []
        state['nonexistent_batch_pid_lists'].append(nonexistent_pids)

        if 'valid_batch_pid_lists' not in state:
            state['valid_batch_pid_lists'] = []

        if pad == False:
            state['valid_batch_pid_lists'].append(valid_pid_list)
            return
        else:
            # get the pids from the batch that are missing from valid_pid_list and append them to the end of valid_pid_list in the order those passages occured.
            remaining_pids = [
                psg["pid"] for psg in state["current_batch"]
                if psg["pid"] not in seen_pids
            ]
            n_pads = len(remaining_pids)
            if n_pads > 0:
                self.logger.warning(f"Padding required: {n_pads}")
            valid_pid_list.extend(remaining_pids)
            state['valid_batch_pid_lists'].append(valid_pid_list)

            if 'padding_batch_pid_lists' not in state:
                state['padding_batch_pid_lists'] = []
            state['padding_batch_pid_lists'].append(remaining_pids)

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

        template_dir = self.template_config["pw_rerank"]
        prompt = self.render_prompt(prompt_dict, template_dir)

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


    def render_prompt(self, prompt_dict, template_dir):
        template = self.jinja_env.get_template(template_dir)
        return template.render(prompt_dict)

    def parse_llm_list_pw(self, llm_output):
        # Try parsing the LLM output using JSON list reader
        try:
            return json.loads(llm_output)
        except json.JSONDecodeError:
            #self.logger.debug(f"Could not parse LLM output as list using regex parsing to look for list in LLM output: {llm_output}")
            try:
                match = re.search(r'\[.*?\]', llm_output, re.DOTALL)
                if match:
                    # Extract and convert single-quoted strings to double quotes for JSON compatibility
                    extracted_list = match.group(0).replace("'", '"')
                    return json.loads(extracted_list)
                else:
                    self.logger.warning(f"No valid regex list found in LLM output: {llm_output}")
                    return []
            except Exception as e:
                self.logger.warning(f"Regex extraction failed to parse as JSON: {e}")
                return []
        except Exception as e:
            self.logger.warning(f"No valid list found in LLM output: {e}")
            return []
        
    def parse_llm_list_lw(self, llm_output):
        try:
            match = re.search(r'\[([^\]]+)\]', llm_output)
            if match:
                extracted_list = match.group(1)
                items = [item.strip().strip('"').strip("'") for item in extracted_list.split(',')]
                # Prepend "p" to items that do not start with "p", e,g, "2" becomes "p2"
                items = [item if item.startswith("p") else f"p{item}" for item in items]
                return items
            else:
                self.logger.warning(f"No valid regex list found in LLM output: {llm_output}")
                return []
        except Exception as e:
            self.logger.warning(f"Error extracting list: {e}")
            return []
    
    def get_local_psgs(self, psg_list):        
        p_ids = [psg["pid"] for psg in psg_list]
        p_texts = [psg["text"] for psg in psg_list]
        if self.config['data'].get('use_simple_pids', False):
            local_psgs = {f"p{i+1}": {"pid": p_id, "text": p_texts[i]} for i, p_id in enumerate(p_ids)}
            return local_psgs
        local_psgs = {p_id: {"pid": p_id, "text": p_texts[i]} for i, p_id in enumerate(p_ids)}
        return local_psgs

if __name__ == "__main__":
    #temporary testing for prompter
    parser = argparse.ArgumentParser(description="Test the Prompter class.")
    parser.add_argument("-c", "--config_path", type=str, help="The path to the config file.")
    args = parser.parse_args()

    with open(args.config_path, "r") as config_file:
        config = yaml.safe_load(config_file)

    load_dotenv()

    prompter = Prompter(config)

    # Example toy state for testing
    state = {
        "instance": {
            "query": {
                "qid": 47923,
                "text": "axon terminals or synaptic knob definition"
            },
            "psg_list": [
                {
                    "pid": "5032362",
                    "text": "What is the term used to describe the rounded areas on the ends of the axon terminals? a) synaptic vesicles Incorrect. Synaptic vesicles are structures within the synaptic knobs. b) axons c) dendrites d) synaptic knobs Correct. Synaptic knobs are located at the tip of each axon terminal."
                },
                {
                    "pid": "1681334",
                    "text": "Psychology Definition of TERMINAL BUTTON: the terminal part of an axon from which a neural signal is rendered, via dispersion of a neurotransmitter, across a synapse to a nearby neuron TERMINAL BUTTON: The terminal button is commonly referred to as the synaptic button, end button, button terminal, terminal bulb, and synaptic knob. Related Psychology Terms axon terminal"
                },
                {
                    "pid": "1868437",
                    "text": "The dendrites don\u00e2\u0080\u0099t have terminating knobs at the end of it. The axons are what conduct action potentials away from the cell body. The axon is not an axon because it\u00e2\u0080\u0099s long, it\u00e2\u0080\u0099s an axon because of the existence of synaptic knobs at the axon terminals. When the axon conducts action potentials away from the cell body and the signal goes to the synaptic knob, it\u00e2\u0080\u0099s going to release a neurotransmitter in response to the electrical signal."
                },
                {
                    "pid": "8641107",
                    "text": "Axons usually have thousands of terminal branches that each end as a bulbous enlargement called a synaptic knob or synaptic terminal. Synaptic knobs contain several membrane-bounded synaptic vesicles that are 40 to 100 nanometers in diameter."
                },
                {
                    "pid": "8418681",
                    "text": "Axons often have thousands of terminal branches, each ending as a bulbous enlargement, the synaptic knob or synaptic terminal. At the synaptic knob, the action potential is converted into a chemical message which, in turn, interacts with the recipient neuron or effector. This process is synaptic transmission."
                }
            ]
        },
        "current_batch": [
            {
                "pid": "5032362",
                "text": "What is the term used to describe the rounded areas on the ends of the axon terminals? a) synaptic vesicles Incorrect. Synaptic vesicles are structures within the synaptic knobs. b) axons c) dendrites d) synaptic knobs Correct. Synaptic knobs are located at the tip of each axon terminal."
            },
            {
                "pid": "1681334",
                "text": "Psychology Definition of TERMINAL BUTTON: the terminal part of an axon from which a neural signal is rendered, via dispersion of a neurotransmitter, across a synapse to a nearby neuron TERMINAL BUTTON: The terminal button is commonly referred to as the synaptic button, end button, button terminal, terminal bulb, and synaptic knob. Related Psychology Terms axon terminal"
            },
            {
                "pid": "1868437",
                "text": "The dendrites don\u00e2\u0080\u0099t have terminating knobs at the end of it. The axons are what conduct action potentials away from the cell body. The axon is not an axon because it\u00e2\u0080\u0099s long, it\u00e2\u0080\u0099s an axon because of the existence of synaptic knobs at the axon terminals. When the axon conducts action potentials away from the cell body and the signal goes to the synaptic knob, it\u00e2\u0080\u0099s going to release a neurotransmitter in response to the electrical signal."
            },
            {
                "pid": "8641107",
                "text": "Axons usually have thousands of terminal branches that each end as a bulbous enlargement called a synaptic knob or synaptic terminal. Synaptic knobs contain several membrane-bounded synaptic vesicles that are 40 to 100 nanometers in diameter."
            },
            {
                "pid": "8418681",
                "text": "Axons often have thousands of terminal branches, each ending as a bulbous enlargement, the synaptic knob or synaptic terminal. At the synaptic knob, the action potential is converted into a chemical message which, in turn, interacts with the recipient neuron or effector. This process is synaptic transmission."
            }
        ]
    }

    # Test the pw_rerank function
    prompter.lw_rerank(state)
    prompter.lw_rerank(state)
    print("Updated state returned:", state)
    
    #testing padding/hallucinations/duplicates
    #state = {
    #    "current_batch": [
    #        {"pid": "p1", "text": "Passage 1"},
    #        {"pid": "p2", "text": "Passage 2"},
    #        {"pid": "p3", "text": "Passage 3"}
    #    ]
    #}
    #raw_batch_pids = ["p1", "p4", "p1"]
    #local_psgs = {
    #    "p1": {"pid": "p1", "text": "Passage 1"},
    #    "p2": {"pid": "p2", "text": "Passage 2"},
    #    "p3": {"pid": "p3", "text": "Passage 3"}
    #}
    #prompter.update_lw_pid_outputs(state, raw_batch_pids, local_psgs, pad=True)
    #print("Updated state:", state)

