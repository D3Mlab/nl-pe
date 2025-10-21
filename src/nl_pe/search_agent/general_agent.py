from nl_pe.utils.setup_logging import setup_logging
from nl_pe.search_agent.base_agent import BaseAgent

#import copy

class GeneralAgent(BaseAgent):

    def __init__(self, config):
        super().__init__(config)

        from nl_pe.search_agent.registry import POLICY_CLASSES
        self.policy_class = POLICY_CLASSES.get(self.agent_config.get('policy'))
        self.policy = self.policy_class(self.config)

    def act(self, query: str, qid: str) -> dict:

        self.curr_state = {
            "query": query,
            'qid': qid,
            'terminate': False
            }

        while True:
            #get next_action(state) or None if no next action
            next_action = self.policy.next_action(self.curr_state)

            if not next_action:
                return self.curr_state

            #next action updates the state
            next_action(self.curr_state)


