from abc import ABC, abstractmethod
import os
import time
from openai import OpenAI
from dotenv import load_dotenv
import yaml
from google import genai
from google.genai import types
from google.api_core.exceptions import ResourceExhausted
#import google.generativeai as genai
#from google.generativeai.types import HarmCategory, HarmBlockThreshold
#from google.api_core.exceptions import GoogleAPICallError, RetryError, InvalidArgument
import random
import google
import argparse
import json
import re
#from botocore.config import Config
from typing import Optional
from nl_pe.utils.setup_logging import setup_logging

class BaseLLM(ABC):

    def __init__(self,config, model_name = ""):
        #model_name: e.g. gpt-3.5-turbo-1106

        self.config = config
        self.logger = setup_logging(self.__class__.__name__, self.config)

        self.model_name = model_name

        self.llm_config = self.config.get('llm', {})
        self.dwell_time = self.llm_config.get('dwell_time', 2) 
        self.num_retries = self.llm_config.get('num_retries', 40)

        self.temp = self.llm_config.get('temperature', 1.0)
        
        
    def prompt(self, *args, **kwargs):
        """Prompts with retries and catches errors"""
        attempt = 0
        while attempt < self.num_retries:
            try:
                self.logger.debug("calling llm api")
                response = self.call_api(*args, **kwargs)
                # Parse JSON from the response message after successful API call
                self._add_json_parsing_to_response(response)
                return response
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

    def _add_json_parsing_to_response(self, response):
        """Parse JSON from LLM response and add to response dict"""
        if "message" not in response:
            return

        message = response["message"]
        parsed_json = self._parse_llm_json(message)

        if parsed_json and isinstance(parsed_json, dict):
            response["JSON_dict"] = parsed_json
            self.logger.debug("Successfully parsed LLM response as JSON object")
        else:
            response["JSON_dict"] = None
            self.logger.warning("JSON parsing failure - response was not valid JSON or not a JSON object")

    def _parse_llm_json(self, llm_output):
        """
        Parse JSON from LLM output with fallback strategies.

        Args:
            llm_output (str): The raw output from the LLM

        Returns:
            dict or None: Parsed JSON object if successful, None if parsing fails
        """
        # Try parsing the LLM output as direct JSON first
        try:
            parsed = json.loads(llm_output)
            # Only return if it's a dictionary object
            if isinstance(parsed, dict):
                return parsed
            return None
        except json.JSONDecodeError:
            # Fallback to regex parsing to find JSON-like structures
            try:
                # Look for object patterns: {key: value, ...}
                match = re.search(r'\{.*?\}', llm_output, re.DOTALL)
                if match:
                    extracted_json = match.group(0)
                    # Convert single-quoted strings to double quotes for JSON compatibility
                    extracted_json = extracted_json.replace("'", '"')
                    # Handle common JSON formatting issues
                    extracted_json = re.sub(r',\s*}', '}', extracted_json)  # Remove trailing commas
                    extracted_json = re.sub(r',\s*]', ']', extracted_json)  # Remove trailing commas

                    parsed = json.loads(extracted_json)
                    # Only return if it's a dictionary object
                    if isinstance(parsed, dict):
                        return parsed
                return None
            except Exception as e:
                self.logger.debug(f"Failed to parse LLM output as JSON: {e}")
                return None
        except Exception as e:
            self.logger.debug(f"Failed to parse LLM output as JSON: {e}")
            return None

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
        self.client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

    def call_api(self, prompt):
        start_time = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temp,
            response_format={ "type": "json_object" }
        )
        end_time = time.perf_counter()
        duration = end_time - start_time

        # Extract token usage from response
        usage = response.usage
        input_tokens = getattr(usage, 'prompt_tokens', 0)
        output_tokens = getattr(usage, 'completion_tokens', 0)

        return {
            "message": response.choices[0].message.content,
            "prompt_time": duration,
            "prompt_tokens": input_tokens,
            "output_tokens": output_tokens
        }
    
class GeminiLLM(BaseLLM):
    def __init__(self, config, model_name):
        super().__init__(config, model_name)

        # Uses GOOGLE_API_KEY from environment automatically
        self.client = genai.Client()

    def call_api(self, prompt):
        start_time = time.perf_counter()

        safety_settings = [
            types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT",
                threshold="BLOCK_NONE",
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH",
                threshold="BLOCK_NONE",
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                threshold="BLOCK_NONE",
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="BLOCK_NONE",
            ),
        ]

        config = types.GenerateContentConfig(
            temperature=self.temp,
            safety_settings=safety_settings,
            response_mime_type="application/json",
        )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config,
        )

        duration = time.perf_counter() - start_time

        usage = getattr(response, "usage_metadata", None)

        return {
            "message": response.text,
            "prompt_time": duration,
            "prompt_tokens": getattr(usage, "prompt_token_count", None),
            "output_tokens": getattr(usage, "candidates_token_count", None),
        }



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prompt an LLM based on a config file.")
    parser.add_argument("-c", "--config_path", type=str, help="The path to the config file.")
    parser.add_argument("-p", "--prompt", type=str, help="The prompt to send to the LLM.")
    args = parser.parse_args()

    with open(args.config_path, "r") as config_file:
        config = yaml.safe_load(config_file)

    load_dotenv()
    #llm = NovaLLM(config, config.get('llm', {}).get('model_name'))
    llm = GeminiLLM(config, config.get('llm', {}).get('model_name'))

    response = llm.prompt(args.prompt)
    print(f'LLM response: {response}')
