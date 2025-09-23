from nl_pe.utils.setup_logging import setup_logging
from nl_pe.search_agent.base_agent import BaseAgent
from nl_pe.utils.utils import AgentLogic
#import types
import copy

class GeneralAgent(BaseAgent):

    def __init__(self, config):
        super().__init__(config)

        from .registry import POLICY_CLASSES
        self.policy_class = POLICY_CLASSES.get(self.agent_config.get('policy'))


    def rank(self, instance):
        #instance: {query: {qid: __, text: __}, psg_list = [{pid: __, text: __},...]}
        self.policy = self.policy_class(self.config)
        self.curr_state = {
            "instance": instance,
            "iteration": 0,
            'preprocessing_done': False,
            'terminate': False
            }

        while True:
            #get next_action(state) or None if no next action
            next_action = self.policy.next_action(self.curr_state)

            if not next_action:
                #return final agent state:
                return self.curr_state

            #next action updates the state
            next_action(self.curr_state)


