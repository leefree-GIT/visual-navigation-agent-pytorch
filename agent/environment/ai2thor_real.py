import time
from agent.environment.environment import Environment
import numpy as np

import torch.multiprocessing as mp
from torchvision import transforms
import torchvision.transforms.functional as F
import torch
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
class THORDiscreteEnvironment(Environment):
    '''
    name:
        Can be one of those:
            # Kitchens:       FloorPlan1 - FloorPlan30
            # Living rooms:   FloorPlan201 - FloorPlan230
            # Bedrooms:       FloorPlan301 - FloorPlan330
            # Bathrooms:      FloorPLan401 - FloorPlan430
            # FloorPlan1
    '''
    def __init__(self, 
                name = "FloorPlan1", 
                grid_size = 0.25,
                input_queue: mp.Queue = None,
                output_queue: mp.Queue = None,
                evt: mp.Event = None,
                history_length: int = 4,
                 **kwargs):
        import ai2thor.controller
        self.name = name
        self.grid_size = grid_size
        self.controller = ai2thor.controller.Controller()
        self.state = None
        self.i_queue = input_queue
        self.o_queue = output_queue
        self.evt = evt
        self.started = False   
        
        self.history_length = history_length

        self.transform = transforms.Compose([
        transforms.Resize((224,224)), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        self.time = 0
        self.terminal = False
        self.collided = False

    def start(self):
        if not self.started:
            self.controller.start() 
            self.started = True
            self.reset()
        self.time = 0
        self.terminal = False
        self.collided = False

    def reset(self):
        # Reset will start the controller (unity window)
        self.start()

        # Reset the environnment
        self.controller.reset(self.name)

        # gridSize specifies the coarseness of the grid that the agent navigates on
        self.state = self.controller.step(dict(action='Initialize', gridSize=self.grid_size, renderObjectImage=True))
        self.s_t = self._tiled_state(self.state.frame)
        

    def _tiled_state(self, frame):
        f = self._get_state(frame)
        return np.tile(f, (1, self.history_length))
    
    def _get_state(self, frame):
        frame_tensor = torch.from_numpy(frame.copy())
        frame_tensor = self.transform(F.to_pil_image(frame_tensor))
        frame_tensor = frame_tensor.unsqueeze(0)
        self.o_queue.put(frame_tensor)
        self.evt.set()
        output = self.i_queue.get()
        return output

    def step(self, action=0):
        assert not self.terminal
        self.state = self.controller.step(dict(action=self.actions[action]))
        self.s_t = np.append(self.s_t[:,1:], self._get_state(self.state.frame), axis=1)

        self.time = self.time + 1

    def render(self, mode = "rgb_array"):
        if mode == "rgb_array":
            return self.state.frame
        else:
            return self.s_t

    def render_target(self, mode = "rgb_array"):
        if mode == "rgb_array":
            return self.state
        else:
            return self.s_t #TODO return target

    @property
    def boudingbox(self):
        return self.state.instance_detections2D
        
    @property
    def actions(self):
        return ["MoveAhead", "RotateRight", "RotateLeft", "MoveBack"]

    @property
    def is_terminal(self):
        return self.terminal or self.time >= 5e3

def make(name):
    if name == "unity":
        return THORDiscreteEnvironment()

if __name__ == '__main__':
    AI2ThorEnv = make('unity')
    AI2ThorEnv.start()
    while True:
        print(AI2ThorEnv.boudingbox)
        AI2ThorEnv.step()
        time.sleep(1)
