import json
import re
import jinja2
from nl_pe.utils.setup_logging import setup_logging
from nl_pe.utils.utils import *
from nl_pe.utils.scorer import TextScorer
from nl_pe.llm import LLM_CLASSES
import argparse
import yaml
from dotenv import load_dotenv

class Prompter(TextScorer):
    
    def __init__(self,config):
        super().__init__(config)

        self.llm_config = config.get('llm', {}) 
        self.model_class_name = self.llm_config.get('model_class')
        self.model_name = self.llm_config.get('model_name')
        self.template_config = config.get("templates", {})
        self.template_dir = self.template_config.get('template_dir')

        model_class = LLM_CLASSES.get(self.model_class_name)
        self.llm = model_class(config,self.model_name)
        self.jinja_env = jinja2.Environment(loader=jinja2.FileSystemLoader(searchpath=self.template_dir))

        self.pw_prompt_path = None
        self.lw_prompt_path = None

        if self.template_config.get('lw_prompt'):
            self.lw_prompt_path = f"{self.template_config.get('lw_prompt')}.jinja2"
        if self.template_config.get('pw_prompt'):
            self.pw_prompt_path = f"{self.template_config.get('pw_prompt')}.jinja2"

        #hardcode query_rel_label for pw_umb since it uses a 0-3 scale instead of 0-1
        if self.pw_prompt_path == "pw_umb.jinja2":
                self.llm_max_label = 3

    def score(self, state, doc_ids):
        state.setdefault("observation_times", [])
        #add scoring prompt to cache name if it doesn't end with it
        self.logger.debug(f"PW LLM Scoring doc_ids: {doc_ids}")
        #track how often cache is used
        cache_use_counter = state.setdefault("cache_use_counter", {})
        cache_use_counter.setdefault("used", 0)
        cache_use_counter.setdefault("not_used", 0)

        # exact-match cache key from full doc_ids sequence
        cache_key = "-".join([str(did) for did in doc_ids])

        # check cache (exact match only)
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            cached_scores = [float(s) for s in entry["scores"]]
            cached_time = float(entry["time"])
            cached_prompt_tokens = entry.get("prompt_tokens", None)

            state["observation_times"] += [cached_time]
            if 'prompt_tokens' not in state:
                state['prompt_tokens'] = []
            state['prompt_tokens'].append(cached_prompt_tokens)

            cache_use_counter["used"] += 1
            self.logger.debug("PW cache hit for doc_ids key: %s", cache_key)
            return cached_scores

        # if not cache:
        cache_use_counter["not_used"] += 1

        query_text = state.get("query")
        d_texts = self.did_to_text(state, doc_ids)
        local_p_ids = [f"p{i+1}" for i in range(len(doc_ids))]

        prompt_dict = {
            'query' : query_text,
            "local_p_ids": local_p_ids,
            "p_texts": d_texts,
            "list_len": len(doc_ids),
        }

        prompt = self.render_prompt(prompt_dict, self.pw_prompt_path)

        self.logger.debug(f"Using template: {self.pw_prompt_path}")
        self.logger.debug("Rendered prompt:\n%s", prompt)

        llm_response = self.llm.prompt(prompt)
        #llm_output = llm_response["message"]
        self.logger.debug("Raw LLM response:\n%s", llm_response)

        int_scores = self._parse_scores_from_JSON(llm_response, len(doc_ids))
        self.logger.debug(f"Parsed scores: {int_scores}")

        #scale scores to 0-1 then mult by self.query_rel_label 
        scale = self.query_rel_label / self.llm_max_label
        scores = [s * scale for s in int_scores]

        # write exact-match cache entry (store scaled scores)
        self.cache[cache_key] = {
            "scores": [float(s) for s in scores],
            "time": float(llm_response["prompt_time"]),
            "prompt_tokens": llm_response.get("prompt_tokens", None),
        }

        state["observation_times"] += [llm_response["prompt_time"]]

        if 'prompt_tokens' not in state:
            state['prompt_tokens'] = []
        state['prompt_tokens'].append(llm_response.get('prompt_tokens', None))

        return scores
    
    def lw_rerank(self, state, doc_ids):
        #doc_ids are the ids in a window. need to return a re-ordering
        state.setdefault("observation_times", [])
        self.logger.debug(f"LW LLM Reranking doc_ids: {doc_ids}")
        #track how often cache is used
        cache_use_counter = state.setdefault("cache_use_counter", {})
        cache_use_counter.setdefault("used", 0)
        cache_use_counter.setdefault("not_used", 0)
        
        # exact-match cache key from full doc_ids sequence
        cache_key = "-".join([str(did) for did in doc_ids])

        # check cache (exact match only)
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            cached_list = entry.get("reranked_list", None)
            cached_time = float(entry.get("time", 0.0))
            cached_prompt_tokens = entry.get("prompt_tokens", None)

            if cached_list is not None:
                state["observation_times"] += [cached_time]
                if "prompt_tokens" not in state:
                    state["prompt_tokens"] = []
                state["prompt_tokens"].append(cached_prompt_tokens)

                cache_use_counter["used"] += 1
                self.logger.debug("LW cache hit for doc_ids key: %s", cache_key)
                return cached_list
            else:
                # Defensive: cache entry exists but doesn't have expected field
                self.logger.warning("LW cache entry missing 'reranked_list' for key: %s", cache_key)

        cache_use_counter["not_used"] += 1

        query_text = state.get("query")
        d_texts = self.did_to_text(state, doc_ids)
        local_p_ids = [f"p{i+1}" for i in range(len(doc_ids))]

        prompt_dict = {
            'query' : query_text,
            "local_p_ids": local_p_ids,
            "p_texts": d_texts,
            "list_len": len(doc_ids),
        }

        prompt = self.render_prompt(prompt_dict, self.lw_prompt_path)

        self.logger.debug(f"Using template: {self.lw_prompt_path}")
        self.logger.debug("Rendered prompt:\n%s", prompt)

        llm_response = self.llm.prompt(prompt)
        self.logger.debug("Raw LLM response:\n%s", llm_response)

        #parsed, deduplicated, hallucinations removed, extended w original order if too short
        reranked_loc_pids  = self._parse_list_from_JSON(llm_response, local_p_ids)
        self.logger.debug(f"Parsed list: {reranked_loc_pids}")

        loc_pid_to_pid = {
            loc_pid: pid for loc_pid, pid in zip(local_p_ids, doc_ids)
        }

        reranked_pids = [
            loc_pid_to_pid[loc_pid] for loc_pid in reranked_loc_pids
        ]

        # write exact-match cache entry (store scaled scores)
        self.cache[cache_key] = {
            "reranked_list": reranked_pids,
            "time": float(llm_response["prompt_time"]),
            "prompt_tokens": llm_response.get("prompt_tokens", None),
        }

        state["observation_times"] += [llm_response["prompt_time"]]

        if 'prompt_tokens' not in state:
            state['prompt_tokens'] = []
        state['prompt_tokens'].append(llm_response.get('prompt_tokens', None))

        return reranked_pids




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

    def _parse_scores_from_JSON(self, llm_response, expected_len):
        """
        Extract scores from llm_response["JSON_dict"].

        Expected structure:
            {"scores": [1,2,3]}
        or
            {"scores": "[1,2,3]"}
        """

        json_dict = llm_response.get("JSON_dict")

        if not json_dict:
            raise ValueError("LLM response did not contain valid JSON_dict.")

        if "scores" not in json_dict:
            raise ValueError("JSON response missing 'scores' key.")

        raw_scores = json_dict["scores"]

        # Case 1: Already a list
        if isinstance(raw_scores, list):
            scores = raw_scores

        # Case 2: String representation of list
        elif isinstance(raw_scores, str):
            try:
                scores = json.loads(raw_scores)
            except Exception:
                # fallback: try python literal parsing
                scores = eval(raw_scores)

        else:
            raise ValueError(f"'scores' must be list or string, got {type(raw_scores)}")

        if not isinstance(scores, list):
            raise ValueError("Parsed scores is not a list.")

        if len(scores) != expected_len:
            raise ValueError(
                f"Score length mismatch: expected {expected_len}, got {len(scores)}"
            )

        try:
            scores = [int(s) for s in scores]
        except Exception:
            raise ValueError("Scores could not be cast to integers.")

        return scores

    def _parse_list_from_JSON(self, llm_response, expected_passage_ids):
        """
        Parse a listwise reranking from llm_response["JSON_dict"].

        Expected JSON structure:
            {"reranked_passage_ids": ["pid1", ..., "pidK"]}
        But "reranked_passage_ids" may also be:
            - a single string like "pid1, pid2, ..., pidK"
            - a list containing one string like ["pid1, pid2, ..., pidK"]
            - a string with brackets/quotes like "['pid1','pid2',...]" or '["pid1","pid2",...]'

        Behavior:
        - If fewer than K ids are returned, log a warning and pad missing ids at the end
            in their original relative order (from expected_passage_ids).
        - If extra/unknown ids appear, log a warning and ignore unknowns.
        - Always returns a list of length K (after padding), unless expected_passage_ids is empty.
        """
        K = len(expected_passage_ids)
        if K == 0:
            return []

        json_dict = llm_response.get("JSON_dict")
        if json_dict is None or not isinstance(json_dict, dict):
            self.logger.warning("LLM response did not contain a valid JSON_dict dict -- returning original ordering")
            return expected_passage_ids

        if "reranked_passage_ids" not in json_dict:
            self.logger.warning("JSON response missing 'reranked_passage_ids' key -- returning original ordering")
            return expected_passage_ids

        raw = json_dict["reranked_passage_ids"]

        def _splitish(s: str):
            # Remove wrapping whitespace and common wrappers
            s = s.strip()
            # Strip a single pair of outer brackets/parentheses if present
            if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
                s = s[1:-1].strip()

            # Remove quotes around whole string if present
            if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
                s = s[1:-1].strip()

            # Now split on commas primarily; fall back to whitespace if no commas.
            parts = [p.strip() for p in s.split(",")] if "," in s else s.split()
            # Strip quotes around each token
            cleaned = []
            for p in parts:
                p = p.strip()
                if not p:
                    continue
                if (p.startswith('"') and p.endswith('"')) or (p.startswith("'") and p.endswith("'")):
                    p = p[1:-1].strip()
                if p:
                    cleaned.append(p)
            return cleaned

        # Normalize to a list of candidate ids (strings)
        candidates = []
        if isinstance(raw, list):
            # e.g. ["pid1","pid2"] OR ["pid1, pid2, pid3"]
            for item in raw:
                if item is None:
                    continue
                if isinstance(item, str):
                    # If item looks like it contains multiple ids, split it.
                    if "," in item or item.strip().startswith("[") or item.strip().startswith("("):
                        candidates.extend(_splitish(item))
                    else:
                        candidates.append(item.strip().strip('"').strip("'"))
                else:
                    # Non-string list entries: coerce to str cautiously
                    candidates.append(str(item).strip())
        elif isinstance(raw, str):
            candidates = _splitish(raw)
        else:
            raise ValueError(f"'reranked_passage_ids' must be list or string, got {type(raw)}")

        expected_set = set(expected_passage_ids)

        # Filter to expected ids, de-dup, preserve order
        seen = set()
        reranked = []
        unknown = []
        for pid in candidates:
            if pid in expected_set:
                if pid not in seen:
                    seen.add(pid)
                    reranked.append(pid)
            else:
                unknown.append(pid)

        if unknown:
            self.logger.warning(
                "LW JSON contained unknown passage ids (ignored): %s", unknown
            )

        # Pad missing ids at end in original relative order
        missing = [pid for pid in expected_passage_ids if pid not in seen]
        if missing:
            self.logger.warning(
                "LW JSON returned %d/%d passage ids; padding %d missing ids at end.",
                len(reranked), K, len(missing)
            )
            reranked.extend(missing)

        # If model returned more than K valid ids (shouldn't happen after de-dup, but just in case)
        if len(reranked) > K:
            self.logger.warning(
                "LW JSON returned more than %d ids after normalization; truncating extras.", K
            )
            reranked = reranked[:K]

        return reranked

if __name__ == "__main__":
    #temporary testing for prompter
    parser = argparse.ArgumentParser(description="Test the Prompter class.")
    parser.add_argument("-c", "--config_path", type=str, help="The path to the config file.")
    args = parser.parse_args()

    with open(args.config_path, "r") as config_file:
        config = yaml.safe_load(config_file)

    load_dotenv()

    prompter = Prompter(config)
