# registry.py
from nl_pe.search_agent.general_agent import GeneralAgent
from nl_pe.llm.prompter import Prompter
from nl_pe.search_agent.policies import PipelinePolicy
from nl_pe.utils.utils import AgentLogic

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