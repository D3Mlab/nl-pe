
import os
import yaml
from dotenv import load_dotenv
from nl_pe.llm import LLM_CLASSES
from nl_pe.llm.prompter import Prompter
from pathlib import Path

#tester code for llms
#should run as module: >python -m llm
if __name__ == "__main__":

    config_path = Path('configs/llm/config.yaml')
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)

    load_dotenv()

    prompter = Prompter(config)

    init_state = {"queries": ["email your doctor from Hillcrest Clinics", "email your doctor contact details Hillcrest Clinics"], "curr_top_k_docIDs": ["2506722","1331194"], "retrieved_lists" : [["1331194", "2697809"],["1331194", "2697809"]]}

    new_state = prompter.rerank_best_and_latest(init_state)

    print('new_state: ', new_state)

    #TO RUN LLM:

    #llm_class = LLM_CLASSES.get(model_class)

    #llm = llm_class(config, model_name)
    #prompt = "2+2="
    #response = llm.prompt(prompt)

    #print(response)
