from abc import ABC, abstractmethod
from llm_passage_ranking.utils.setup_logging import setup_logging

class BaseAgent(ABC):

    def __init__(self, config):
        
        self.config = config
        self.agent_config = self.config.get('agent', {})
        self.logger = setup_logging(self.__class__.__name__, self.config)

    @abstractmethod
    def rank(self, query: str) -> dict:
         raise NotImplementedError("This method must be implemented by a subclass.")
