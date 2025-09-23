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


    def prompt(self, prompt):
        """
        Generic method to call the LLM with a prompt and return just the text response.

        Args:
            prompt (str): The prompt string to send to the LLM

        Returns:
            str: The text response from the LLM
        """
        llm_response = self.llm.prompt(prompt)
        return llm_response["message"]

    def render_prompt(self, prompt_dict, template_dir):
        template = self.jinja_env.get_template(template_dir)
        return template.render(prompt_dict)

    def parse_llm_json(self, llm_output, add_p_prefix=False):
        """
        Generic method to parse any JSON output from LLM, with optional passage ID prefixing.

        Args:
            llm_output (str): The raw output from the LLM
            add_p_prefix (bool): Whether to add "p" prefix to items that don't start with "p"

        Returns:
            list: Parsed list from the LLM output, or empty list if parsing fails
        """
        # Try parsing the LLM output as direct JSON first
        try:
            return json.loads(llm_output)
        except json.JSONDecodeError:
            # Fallback to regex parsing to find JSON-like structures
            try:
                # Look for array patterns: [item1, item2, ...]
                match = re.search(r'\[.*?\]', llm_output, re.DOTALL)
                if match:
                    extracted_json = match.group(0)
                    # Convert single-quoted strings to double quotes for JSON compatibility
                    extracted_json = extracted_json.replace("'", '"')
                    # Handle common JSON formatting issues
                    extracted_json = re.sub(r',\s*}', '}', extracted_json)  # Remove trailing commas
                    extracted_json = re.sub(r',\s*]', ']', extracted_json)  # Remove trailing commas

                    parsed_list = json.loads(extracted_json)

                    # Apply p prefixing if requested
                    if add_p_prefix and isinstance(parsed_list, list):
                        parsed_list = [item if str(item).startswith("p") else f"p{item}" for item in parsed_list]

                    return parsed_list
                else:
                    self.logger.warning(f"Failed to parse LLM output as JSON: {llm_output}")
                    return []
            except Exception as e:
                self.logger.warning(f"Failed to parse LLM output as JSON: {e}")
                return []
        except Exception as e:
            self.logger.warning(f"Failed to parse LLM output as JSON: {e}")
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
