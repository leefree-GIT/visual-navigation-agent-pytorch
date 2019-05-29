# -*- coding: utf-8 -*-
import sys
import h5py
import json
import numpy as np
import random
import skimage.io
from skimage.transform import resize
from agent.environment.environment import Environment
from agent.constants import USE_RESNET
import torch.multiprocessing as mp
from torchvision import transforms
import torchvision.transforms.functional as F
import torch

class THORDiscreteEnvironment(Environment):
    def __init__(self, 
            scene_name = 'bedroom_04',
            random_start = True,
            n_feat_per_location = 1,
            history_length : int = 4,
            screen_width = 224,
            screen_height = 224,
            terminal_state_id = 0,
            h5_file_path = None,
            input_queue: mp.Queue = None,
            output_queue: mp.Queue = None,
            evt: mp.Event = None,
            **kwargs):
        super(THORDiscreteEnvironment, self).__init__()



        if h5_file_path is None:
            h5_file_path = f"/app/data/{scene_name}.h5"
        elif callable(h5_file_path):
            h5_file_path = h5_file_path(scene_name)

        self.scene = scene_name
            
        self.use_resnet = USE_RESNET

        self.terminal_state_id = terminal_state_id

        self.h5_file = h5py.File(h5_file_path, 'r')

        self.n_feat_per_location = n_feat_per_location

        self.locations = self.h5_file['location'][()]
        self.rotations = self.h5_file['rotation'][()]

        self.history_length = history_length

        self.n_locations = self.locations.shape[0]

        self.terminals = np.zeros(self.n_locations)
        self.terminals[terminal_state_id] = 1
        self.terminal_states, = np.where(self.terminals)

        self.transition_graph = self.h5_file['graph'][()]

        self.shortest_path_distances = self.h5_file['shortest_path_distance'][()]


        self.i_queue = input_queue
        self.o_queue = output_queue
        self.evt = evt
        self.time = 0

        self.transform = transforms.Compose([
        transforms.Resize((224,224)), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])




        # LAST instruction
        self.s_target = self._tiled_state(self.terminal_state_id)


    def reset(self):
        # randomize initial state
        while True:
            k = random.randrange(self.n_locations)
            min_d = np.inf

            # check if target is reachable
            for t_state in self.terminal_states:
                dist = self.shortest_path_distances[k][t_state]
                min_d = min(min_d, dist)

            # min_d = 0  if k is a terminal state
            # min_d = -1 if no terminal state is reachable from k
            if min_d > 0: break
        
        # reset parameters
        self.current_state_id = k # TODO: k
        self.start_state_id = k
        self.s_t = self._tiled_state(self.current_state_id)

        self.collided = False
        self.terminal = False
        self.time = 0

    def step(self, action):
        assert not self.terminal, 'step() called in terminal state'
        k = self.current_state_id
        if self.transition_graph[k][action] != -1:
            self.current_state_id = self.transition_graph[k][action]
            if self.terminals[self.current_state_id]:
                self.terminal = True
                self.collided = False
            else:
                self.terminal = False
                self.collided = False
        else:
            self.terminal = False
            self.collided = True

        self.s_t = np.append(self.s_t[:,1:], self._get_state(self.current_state_id), axis=1)
        self.time = self.time + 1

    def _get_state(self, state_id):
        # read from hdf5 cache
        k = random.randrange(self.n_feat_per_location)
        if not self.use_resnet:
            return self.h5_file['resnet_feature'][state_id][k][:,np.newaxis]
        else:
            obs = self.h5_file['observation'][state_id]
            frame_tensor = torch.from_numpy(obs)
            frame_tensor = self.transform(F.to_pil_image(frame_tensor))
            frame_tensor = frame_tensor.unsqueeze(0)
            self.o_queue.put(frame_tensor)
            self.evt.set()
            return self.i_queue.get()
            # input_tens = input_tens.to(next(self.resnet_trained.parameters()).device)
            # input_tens = input_tens.unsqueeze(0)
            # res = self.resnet_trained((input_tens,))
            # return res.permute(1,0).cpu()

    def _tiled_state(self, state_id):
        f = self._get_state(state_id)
        return np.tile(f, (1, self.history_length))

    def _calculate_reward(self, terminal, collided):
        # positive reward upon task completion
        if terminal: return 10.0
        # time penalty or collision penalty
        return -0.1 if collided else -0.01

    @property
    def reward(self):
        return self._calculate_reward(self.is_terminal, self.collided)

    @property
    def is_terminal(self):
        return self.terminal or self.time >= 5e3

    @property
    def observation(self):
        return self.h5_file['observation'][self.current_state_id]

    def render(self, mode):
        assert mode == 'resnet_features'
        return self.s_t

    def render_target(self, mode):
        assert mode == 'resnet_features'
        return self.s_target

    @property
    def actions(self):
        return ["MoveForward", "RotateRight", "RotateLeft", "MoveBackward"]

    @property
    def shortest_path_distance_start(self):
        return self.shortest_path_distances[self.start_state_id][self.terminal_state_id]