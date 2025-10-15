
from abc import ABC, abstractclassmethod, abstractmethod
from nl_pe.utils.setup_logging import setup_logging

class BasePolicy(ABC):

    def __init__(self, config):
        
        self.config = config
        self.logger = setup_logging(self.__class__.__name__, self.config)

        from .registry import COMPONENT_CLASSES 
        self.COMPONENT_CLASSES = COMPONENT_CLASSES
        self.components = {}

    @abstractmethod
    def next_action(self, state) -> tuple:
    #return the method from the component instance (e.g. rank() from DenseRetriever) or None if no next action
         raise NotImplementedError("This method must be implemented by a subclass.")
    
    def set_component_instance(self, comp_name):
        #check if we've already instantiated the component)
        if comp_name not in self.components:
            #instantiate new component
            self.logger.debug(f'Instantiating component: {comp_name}')
            comp_class = self.COMPONENT_CLASSES.get(comp_name)
            if not comp_class:
                raise ValueError(f"Component class for {comp_name} not found.")
            self.curr_comp_inst = comp_class(config=self.config)
            self.logger.debug(f'Component instance created: {self.curr_comp_inst}')
            self.components[comp_name] = self.curr_comp_inst
        else:
            self.curr_comp_inst = self.components[comp_name]

class PipelinePolicy(BasePolicy):
    #execute steps in a pipeline up to K times 

    def __init__(self, config):
        super().__init__(config)
        #list of policy steps (i.e. components and methods to call)
        self.steps = self.config.get('agent', {}).get('policy_steps', [])
        #step_cnt (lowest level iteration tracking) is reset to 0 after each pipeline iteration
        self.current_step_cnt = 0
        #iteration count (highest level iteration tracking) is incremented after each pipeline iteration
        self.iteration_cnt = 0

        self.max_pipeline_iterations = self.config.get('agent', {}).get('max_pipeline_iterations',1)


    def next_action(self, state):
        if state.get("terminate"):
            return None

        # If all steps are completed and there are pipeline iterations remaining, reset cnts and repeat the pipeline, else return None to terminate
        if self.current_step_cnt >= len(self.steps):
            self.current_step_cnt = 0
            self.iteration_cnt += 1
            #state["iteration"] = self.iteration_cnt

            if self.iteration_cnt >= self.max_pipeline_iterations:
                return None
                
        step = self.steps[self.current_step_cnt]
        self.current_step_cnt += 1

        #class which takes the step e.g. Propmter
        comp_name = step.get('component')
        #get component instance
        self.set_component_instance(comp_name)
        #method to take, e.g. batch_pw_rerank (query, candidates, config)
        self.curr_method_name = step.get('method')

        self.logger.debug(f'next action: {comp_name}.{self.curr_method_name}')

        # Get the method from the component instance, e.g. rank() from DenseRetriever
        return getattr(self.curr_comp_inst, self.curr_method_name)

