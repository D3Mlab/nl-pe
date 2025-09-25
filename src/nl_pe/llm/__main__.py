
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

    # Use hardcoded prompt with prompter's generic prompt method
    #hardcoded_prompt = r"What is 2+2? Response in a JSON format only: { response: <your response> }"
    #response = prompter.prompt_from_str(hardcoded_prompt)

    template_path = 'hotel_gen.jinja2'
    response = prompter.prompt_from_temp(template_path)

    print("Full response:", response)

    # Check if JSON parsing was successful
    if "JSON_dict" in response:
        print("Parsed JSON:", response["JSON_dict"])
    else:
        print("No JSON_dict field - response was not valid JSON")
