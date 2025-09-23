from abc import ABC, abstractmethod
import os
import time
import openai
from dotenv import load_dotenv
import yaml
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from google.api_core.exceptions import GoogleAPICallError, RetryError, InvalidArgument
import random
import google
import argparse
from langchain_core.messages import SystemMessage, HumanMessage
import boto3
import json
from botocore.config import Config
from typing import Optional

from llm_passage_ranking.utils.setup_logging import setup_logging

class BaseLLM(ABC):

    def __init__(self,config, model_name = ""):
        #model_name: e.g. gpt-3.5-turbo-1106

        self.config = config
        self.logger = setup_logging(self.__class__.__name__, self.config)

        self.model_name = model_name

        self.llm_config = self.config.get('llm', {})
        self.dwell_time = self.llm_config.get('dwell_time', 60) 
        self.num_retries = self.llm_config.get('num_retries', 3)

        self.temp = self.llm_config.get('temperature', 1.0)
        
        
    def prompt(self, *args, **kwargs):
        """Prompts with retries and catches errors"""
        attempt = 0
        while attempt < self.num_retries:
            try:
                self.logger.debug("calling llm api")
                return self.call_api(*args, **kwargs)
            except Exception as e:
                self.logger.warning(f"LLM API Error: {e}")
                self.handle_exception(e, attempt)
                attempt += 1
                if attempt < self.num_retries:
                    self.logger.info(f"Retrying in {self.dwell_time} seconds...")
                    time.sleep(self.dwell_time)
                else:
                    self.logger.error("All retry attempts exhausted.")
                    return {"error": str(e)}

    def handle_exception(self, e, attempt):
        """Default exception handler, can be overridden by subclasses."""
        self.logger.info(f"Attempt {attempt + 1} failed: {e}")
        pass



    @abstractmethod
    def call_api(self, prompt):
        #attempt to call llm and return dict of response results
        #e.g. {message: "Hi", logprobs: (2.4,0.4)}
        """Method to be implemented by subclasses to make the actual API call."""
        raise NotImplementedError("This method must be implemented by a subclass.")



class OpenAILLM(BaseLLM):
    def __init__(self, config, model_name):
        super().__init__(config,model_name)
        openai.api_key = os.environ['OPENAI_API_KEY']

    def call_api(self, prompt, logprobs = False, top_logprobs = 3):
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temp,
            logprobs=logprobs,
        )
        return {
            "message": response.choices[0].message['content'],
            "logprobs": response.choices[0].logprobs if logprobs == True else None
        }
    

class NovaLLM(BaseLLM):
    def __init__(self, config, model_name):
        super().__init__(config,model_name)
        self.bedrock_runtime = self.get_boto_client(assumed_role="arn:aws:iam::451191978663:role/service-role/a207918-ml-workspace-PracticalLawxOhJ-prod-use1", runtime=True)
        self.model_id = model_name

    def call_api(self, prompt):
        body = json.dumps({"messages": [{
                                        "role": "user",
                                        "content": [{"text": prompt}]}],
                            "inferenceConfig": {'temperature': self.temp}})
        accept = 'application/json'
        contentType = 'application/json'

        start_time = time.perf_counter()
        response = self.bedrock_runtime.invoke_model(body=body, modelId=self.model_id, accept=accept, contentType=contentType)
        end_time = time.perf_counter()
        duration = end_time - start_time

        # Process and print the response
        result = json.loads(response.get("body").read())
        print(result)
        input_tokens = result["usage"]["inputTokens"]
        message = result['output']['message']['content'][0]['text']

        return {
            "message": message,
            "prompt_time": duration,
            "prompt_tokens": input_tokens  
        }

class ClaudeLLM(BaseLLM):
    def __init__(self, config, model_name):
        super().__init__(config,model_name)
        self.bedrock_runtime = self.get_boto_client(assumed_role="arn:aws:iam::451191978663:role/service-role/a207918-ml-workspace-PracticalLawxOhJ-prod-use1", runtime=True)
        self.model_id = model_name

    def call_api(self, prompt):
        body=json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}],
                    }
                ],
                "temperature": self.temp,
                'max_tokens': 1000000
            }
        )

        start_time = time.perf_counter()
        response = self.bedrock_runtime.invoke_model(modelId=self.model_id,body=body)
        end_time = time.perf_counter()
        duration = end_time - start_time

        # Process and print the response
        result = json.loads(response.get("body").read())
        input_tokens = result["usage"]["input_tokens"]
        output_list = result.get("content", [])
        message = output_list[0]["text"]

        return {
            "message": message,
            "prompt_time": duration,
            "prompt_tokens": input_tokens  
        }


class GeminiLLM(BaseLLM):
    def __init__(self, config, model_name):
        super().__init__(config,model_name)

        self.GEMINI_API_KEYS = [
            value for key, value in sorted(os.environ.items()) if key.startswith('GEMINI_API_KEY_')
        ]
        self.key_exhausted_status = [False] * len(self.GEMINI_API_KEYS)  # False indicates the key is not exhausted
        self.current_key_index = 0
        genai.configure(api_key=self.GEMINI_API_KEYS[self.current_key_index])
        self.model = genai.GenerativeModel(self.model_name)
        self.safety_settings = {HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,}

    def call_api(self, prompt):
        start_time = time.perf_counter()
        generation_config=genai.types.GenerationConfig(
            temperature=self.temp)
        response = self.model.generate_content(prompt,
                                                safety_settings = self.safety_settings,
                                                generation_config=generation_config)
        end_time = time.perf_counter()
        duration = end_time - start_time
        return {
            "message": response.text,
            "prompt_time": duration  
        }


    def handle_exception(self, e, attempt):
        if isinstance(e, google.api_core.exceptions.ResourceExhausted):
            self.logger.warning(f"Quota reached for current Gemini API key.")
            self.key_exhausted_status[self.current_key_index] = True 

            available_keys = [i for i, exhausted in enumerate(self.key_exhausted_status) if not exhausted]
            if available_keys:
                self.current_key_index = random.choice(available_keys)
                self.logger.info(f"Quota reached for current API key. Switching to key index {self.current_key_index}")
                genai.configure(api_key=self.GEMINI_API_KEYS[self.current_key_index])
            else:
                self.logger.error("All API keys have been exhausted.")
                self.key_exhausted_status = [False] * len(self.GEMINI_API_KEYS)  # False indicates the key is not exhausted
                self.current_key_index = 0
        else:
            self.logger.info(f"Attempt {attempt + 1} failed: {e}")
            pass




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prompt an LLM based on a config file.")
    parser.add_argument("-c", "--config_path", type=str, help="The path to the config file.")
    parser.add_argument("-p", "--prompt", type=str, help="The prompt to send to the LLM.")
    args = parser.parse_args()

    with open(args.config_path, "r") as config_file:
        config = yaml.safe_load(config_file)

    load_dotenv()
    llm = NovaLLM(config, config.get('llm', {}).get('model_name'))

    response = llm.prompt(args.prompt)
    print(f'LLM response: {response}')