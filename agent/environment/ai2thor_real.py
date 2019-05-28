import time
from environment import Environment
import numpy as np

class AI2ThorEnvironment(Environment):
    '''
    name:
        Can be one of those:
            # Kitchens:       FloorPlan1 - FloorPlan30
            # Living rooms:   FloorPlan201 - FloorPlan230
            # Bedrooms:       FloorPlan301 - FloorPlan330
            # Bathrooms:      FloorPLan401 - FloorPlan430
            # FloorPlan1
    '''
    def __init__(self, name = "FloorPlan1", grid_size = 0.25, **kwargs):
        import ai2thor.controller
        self.name = name
        self.grid_size = grid_size
        self.controller = ai2thor.controller.Controller()
        self.state = None
        self.history_length = kwargs.get('history_length', 5)

    def start(self):
        self.controller.start() 
        self.reset()

    def reset(self):
        self.controller.reset(self.name)

        # gridSize specifies the coarseness of the grid that the agent navigates on
        self.state = self.controller.step(dict(action='Initialize', gridSize=self.grid_size, renderObjectImage=True))
        # self.s_t = self._tiled_state(self.current_state_id)

    def _tiled_state(self, state_id):
        f = self._get_state(state_id)
        return np.tile(f, (1, self.history_length))
    
    def _get_state(self, state_id):
        input_tens = self.resized_obs_tens[state_id]
        input_tens = input_tens.to(next(self.resnet_trained.parameters()).device)
        input_tens = input_tens.unsqueeze(0)
        res = self.resnet_trained((input_tens,))
        return res.permute(1,0).cpu()

    def step(self, action='MoveAhead'):
        self.state = self.controller.step(dict(action=action))

    def render(self, mode = "rgb_array"):
        if mode == "rgb_array":
            return self.state.frame

    def render_target(self, mode = "rgb_array"):
        if mode == "rgb_array":
            return self.state

    @property
    def boudingbox(self):
        return self.state.instance_detections2D
        
    @property
    def actions(self):
        return ["MoveAhead", "RotateRight", "RotateLeft", "MoveBackward"]

def make(name):
    if name == "unity":
        return AI2ThorEnvironment()

if __name__ == '__main__':
    AI2ThorEnv = make('unity')
    AI2ThorEnv.start()
    while True:
        print(AI2ThorEnv.boudingbox)
        time.sleep(1)
