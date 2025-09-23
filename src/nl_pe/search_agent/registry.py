# registry.py
from llm_passage_ranking.search_agent.general_agent import GeneralAgent
from llm_passage_ranking.llm.prompter import Prompter
from llm_passage_ranking.search_agent.policies import PipelinePolicy
from llm_passage_ranking.utils.utils import AgentLogic

AGENT_CLASSES = {
    'GeneralAgent': GeneralAgent,
}

# Classes that might be used as components in a general agent (QPP, Embedders, LLMs, etc.)
COMPONENT_CLASSES = {
    'AgentLogic': AgentLogic,
    'Prompter': Prompter
    }

POLICY_CLASSES = {
    'PipelinePolicy': PipelinePolicy,
}