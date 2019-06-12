# -*- coding: utf-8 -*-
import sys
import h5py
import json
import numpy as np
import random
import skimage.io
from skimage.transform import resize
from agent.environment.environment import Environment
import torch.multiprocessing as mp
from torchvision import transforms
import torchvision.transforms.functional as F
import torch
import json

class THORDiscreteEnvironment(Environment):
    def __init__(self, 
            scene_name = 'FloorPlan1',
            resnet_trained = None,
            n_feat_per_location = 1,
            history_length : int = 4,
            terminal_state = 0,
            h5_file_path = None,
            action_size : int = 4,
            **kwargs):
        """THORDiscreteEnvironment constructor, it represent a world where an agent evolves
        
        Keyword Arguments:
            scene_name {str} -- Name of the current world (default: {'bedroom_04'})
            resnet_trained {[type]} -- Resnet network used to compute features (default: {None})
            n_feat_per_location {int} -- Number of feature by position in the world (default: {1})
            history_length {int} -- Number of frame to stack so the network take in account previous observations (default: {4})
            terminal_state_id {int} -- Terminal position represented by an ID (default: {0})
            h5_file_path {[type]} -- Path to precomputed world (default: {None})
            input_queue {mp.Queue} -- Input queue to receive resnet features (default: {None})
            output_queue {mp.Queue} -- Output queue to ask for resnet features (default: {None})
            evt {mp.Event} -- Event to tell the GPUThread that there are new data to compute (default: {None})
        """
        super(THORDiscreteEnvironment, self).__init__()



        if h5_file_path is None:
            h5_file_path = f"/app/data/{scene_name}.h5"
        elif callable(h5_file_path):
            h5_file_path = h5_file_path(scene_name)

        self.scene = scene_name

        self.terminal_state = terminal_state

        self.h5_file = h5py.File(h5_file_path, 'r')

        self.n_feat_per_location = n_feat_per_location

        self.history_length = history_length

        self.locations = self.h5_file['location'][()]
        self.rotations = self.h5_file['rotation'][()]

        self.n_locations = self.locations.shape[0]

        self.transition_graph = self.h5_file['graph'][()]

        self.action_size = action_size


        self.time = 0




        # LAST instruction
        terminal_id = -1
        for i, loc in enumerate(self.locations):
            if np.array_equal(loc, list(self.terminal_state['position'].values())):
                if np.array_equal(self.rotations[i], list(self.terminal_state['rotation'].values())):
                    terminal_id = i
                    break
        self.s_target = self._tiled_state(terminal_id)


    def reset(self):
        # randomize initial state
        k = random.randrange(self.n_locations)
        while True:
            # Assure that Z value is 0
            if self.rotations[k][2] == 0:
                break
            k = random.randrange(self.n_locations)
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
            agent_pos = self.locations[self.current_state_id] # NDARRAY
            agent_rot = self.rotations[self.current_state_id][1] # Check only y value

            terminal_pos = self.terminal_state['position'].values() # NDARRAY
            terminal_rot = self.terminal_state['rotation']['y'] # Check only y value

            if np.array_equal(agent_pos, terminal_pos) and np.array_equal(agent_rot, terminal_rot):
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
        return self.h5_file['resnet_feature'][state_id][k][:,np.newaxis]

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

    @property
    def boudingbox(self):
        return self.h5_file['bbox'][self.current_state_id]

    def render(self, mode):
        assert mode == 'resnet_features'
        return self.s_t

    def render_target(self, mode):
        assert mode == 'resnet_features'
        return self.s_target

    @property
    def actions(self):
        acts = ["MoveAhead", "RotateRight", "RotateLeft", "MoveBack", "LookUp", "LookDown", "MoveRight", "MoveLeft"]
        return acts[:self.action_size]

    def stop(self):
        pass