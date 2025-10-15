from nl_pe.utils.setup_logging import setup_logging
import copy

#class with some basic agent actions
class AgentLogic():

    def __init__(self, config):
        self.config = config
        self.logger = setup_logging(self.__class__.__name__, self.config)
