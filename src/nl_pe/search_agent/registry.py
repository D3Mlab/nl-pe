# registry.py
from nl_pe.search_agent.general_agent import GeneralAgent
from nl_pe.search_agent.agent_logic import AgentLogic
from nl_pe.llm.prompter import Prompter
from nl_pe.search_agent.policies import PipelinePolicy
from nl_pe.embedding.embedders import HuggingFaceEmbedderSentenceTransformers
from nl_pe.embedding.embedders import GoogleEmbedder
from nl_pe.acitve_learning.active_learners import GPActiveLearner

AGENT_CLASSES = {
    'GeneralAgent': GeneralAgent,
}

# Classes that might be used as components in a general agent (QPP, Embedders, LLMs, etc.)
COMPONENT_CLASSES = {
    'AgentLogic': AgentLogic,
    'Prompter': Prompter,
    "HuggingFaceEmbedderSentenceTransformers": HuggingFaceEmbedderSentenceTransformers,
    "GoogleEmbedder": GoogleEmbedder,
    "GPActiveLearner": GPActiveLearner,
    }

POLICY_CLASSES = {
    'PipelinePolicy': PipelinePolicy,
}