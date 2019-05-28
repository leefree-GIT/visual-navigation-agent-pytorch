from agent.network import SceneSpecificNetwork, SharedNetwork, ActorCriticLoss, SharedResnet, compare_models
from agent.environment import Environment, THORDiscreteEnvironment
from agent.resnet import resnet50

import torch.nn as nn
from typing import Dict, Collection
import signal
import random
import torch
import h5py
from agent.replay import ReplayMemory, Sample
from collections import namedtuple
import torch.multiprocessing as mp
import torch.nn.functional as F
import numpy as np
import logging
from tensorboardX import SummaryWriter

import sys
import pdb
import os

class ForkablePdb(pdb.Pdb):

    _original_stdin_fd = sys.stdin.fileno()
    _original_stdin = None

    def __init__(self):
        pdb.Pdb.__init__(self, nosigint=True)

    def _cmdloop(self):
        current_stdin = sys.stdin
        try:
            if not self._original_stdin:
                self._original_stdin = os.fdopen(self._original_stdin_fd)
            sys.stdin = self._original_stdin
            self.cmdloop()
        finally:
            sys.stdin = current_stdin

TrainingSample = namedtuple('TrainingSample', ('state', 'policy', 'value', 'action_taken', 'goal', 'R', 'temporary_difference'))


class TrainingThread(mp.Process):
    def __init__(self,
            id : int,
            network : torch.nn.Module,
            saver,
            optimizer,
            scene : str,
            max_step: int,
            resnet_trained: torch.nn.Module,
            obs_preloaded: torch.tensor,
            **kwargs):

        super(TrainingThread, self).__init__()

        # Initialize the environment
        self.env = None
        self.init_args = kwargs
        self.scene = scene
        self.saver = saver
        self.local_backbone_network = SharedNetwork()
        self.id = id

        if obs_preloaded is None:
            self.use_resnet = False
        else:
            self.use_resnet = True

        self.master_network = network
        self.resnet_network = resnet_trained
        self.optimizer = optimizer

        self.exit = mp.Event()
        self.max_step = max_step
        self.local_t = 0
        self.obs_preloaded = obs_preloaded

    def _sync_network(self):
        state_dict = self.master_network.state_dict()
        self.policy_network.load_state_dict(state_dict)

    def _ensure_shared_grads(self):
        for param, shared_param in zip(self.policy_network.parameters(), self.master_network.parameters()):
            if shared_param.grad is not None:
                return 
            shared_param._grad = param.grad 
    
    def get_action_space_size(self):
        return len(self.env.actions)

    def _initialize_thread(self):
        h5_file_path = self.init_args.get('h5_file_path')
        # self.logger = logging.getLogger('agent')
        # self.logger.setLevel(logging.INFO)
        self.init_args['h5_file_path'] = lambda scene: h5_file_path.replace('{scene}', scene)
        if self.use_resnet:
            self.env = THORDiscreteEnvironment(self.scene, resnet_trained = self.resnet_network, obs_preloaded = self.obs_preloaded, **self.init_args)
        else:
            self.env = THORDiscreteEnvironment(self.scene, resnet_trained = None, obs_preloaded = self.obs_preloaded, **self.init_args)
        self.gamma : float = self.init_args.get('gamma', 0.99)
        self.grad_norm: float = self.init_args.get('grad_norm', 40.0)
        entropy_beta : float = self.init_args.get('entropy_beta', 0.01)
        self.max_t : int = self.init_args.get('max_t', 5)# TODO: 5)
        self.local_t = 0
        self.action_space_size = self.get_action_space_size()

        self.criterion = ActorCriticLoss(entropy_beta)
        self.policy_network = nn.Sequential(SharedNetwork(), SceneSpecificNetwork(self.get_action_space_size()))
        # Initialize the episode
        self._reset_episode()
        self._sync_network()

        self.writer = SummaryWriter("runs/log_output",filename_suffix='id_' + str(self.id) + '_')


    def _reset_episode(self):
        self.episode_reward = 0
        self.episode_length = 0
        self.episode_max_q = -np.inf
        self.env.reset()

    def _forward_explore(self):
        # Does the evaluation end naturally?
        is_terminal = False
        terminal_end = False

        results = { "policy":[], "value": []}
        rollout_path = {"state": [], "action": [], "rewards": [], "done": []}


        # Plays out one game to end or max_t
        for t in range(self.max_t):
            state = { 
                "current": self.env.render('resnet_features'),
                "goal": self.env.render_target('resnet_features'),
            }

            x_processed = torch.from_numpy(state["current"])
            goal_processed = torch.from_numpy(state["goal"])

            (policy, value) = self.policy_network((x_processed, goal_processed,))

            if (self.id == 0) and (self.local_t % 100) == 0:
                print(f'Local Step {self.local_t}')

            # Store raw network output to use in backprop
            results["policy"].append(policy)
            results["value"].append(value)

            with torch.no_grad():
                (_, action,) = policy.max(0)
                action = F.softmax(policy, dim=0).multinomial(1).item()
            
            policy = policy.data.numpy()
            value = value.data.numpy()
            
            
            # Makes the step in the environment
            self.env.step(action)

            # Receives the game reward
            is_terminal = self.env.is_terminal

            # ad-hoc reward for navigation
            reward = 10.0 if is_terminal else -0.01

            # Max episode length
            if self.episode_length > 5e3: is_terminal = True

            # Update episode stats
            self.episode_length += 1
            self.episode_reward += reward
            self.episode_max_q = max(self.episode_max_q, np.max(value))

            # clip reward
            reward = np.clip(reward, -1, 1)

            # Increase local time
            self.local_t += 1

            rollout_path["state"].append(state)
            rollout_path["action"].append(action)
            rollout_path["rewards"].append(reward)
            rollout_path["done"].append(is_terminal)

            if is_terminal:
                # TODO: add logging
                print(f"time {self.optimizer.get_global_step()} | thread #{self.id} | scene {self.scene} | target #{self.env.terminal_state_id}")

                print('playout finished')
                print(f'episode length: {self.episode_length}')
                print(f'episode shortest length: {self.env.shortest_path_distance_start}')
                print(f'episode reward: {self.episode_reward}')
                print(f'episode max_q: {self.episode_max_q}')
                
                scene_log = self.scene + '-' + str(self.init_args.get('terminal_state_id', -1))
                step = self.optimizer.get_global_step()
                self.writer.add_scalar(scene_log + '/episode_length', self.episode_length, step)
                self.writer.add_scalar(scene_log + '/max_q', float(self.episode_max_q), step)
                self.writer.add_scalar(scene_log + '/reward', float(self.episode_reward), step)
                self.writer.add_scalar(scene_log + '/learning_rate', float(self.optimizer.scheduler.get_lr()[0]), step)

                terminal_end = True
                self._reset_episode()
                break      

        if terminal_end:
            return 0.0, results, rollout_path
        else:
            x_processed = torch.from_numpy(self.env.render('resnet_features'))
            goal_processed = torch.from_numpy(self.env.render_target('resnet_features'))

            (_, value) = self.policy_network((x_processed, goal_processed,))
            return value.data.item(), results, rollout_path
    
    def _optimize_path(self, playout_reward: float, results, rollout_path):
        policy_batch = []
        value_batch = []
        action_batch = []
        temporary_difference_batch = []
        playout_reward_batch = []


        for i in reversed(range(len(results["value"]))):
            reward = rollout_path["rewards"][i]
            value = results["value"][i]
            action = rollout_path["action"][i]

            playout_reward = reward + self.gamma * playout_reward
            temporary_difference = playout_reward - value.data.item()

            policy_batch.append(results['policy'][i])
            value_batch.append(results['value'][i])
            action_batch.append(action)
            temporary_difference_batch.append(temporary_difference)
            playout_reward_batch.append(playout_reward)
        
        policy_batch = torch.stack(policy_batch, 0)
        value_batch = torch.stack(value_batch, 0)
        action_batch = torch.from_numpy(np.array(action_batch, dtype=np.int64))
        temporary_difference_batch = torch.from_numpy(np.array(temporary_difference_batch, dtype=np.float32))
        playout_reward_batch = torch.from_numpy(np.array(playout_reward_batch, dtype=np.float32))
        
        # Compute loss
        loss = self.criterion.forward(policy_batch, value_batch, action_batch, temporary_difference_batch, playout_reward_batch)
        loss = loss.sum()

        loss_value = loss.detach().numpy()
        self.optimizer.optimize(loss, 
            self.policy_network.parameters(), 
            self.master_network.parameters())

    def run(self, master = None):
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        print(f'Thread {self.id} ready')
        
        # We need to silence all errors on new process
        h5py._errors.silence_errors()
        self._initialize_thread()
                
        if not master is None:
            print(f'Master thread {self.id} started')
        else:
            print(f'Thread {self.id} started')

        try:
            self.env.reset()
            while True and not self.exit.is_set() and self.optimizer.get_global_step() < self.max_step:
                self._sync_network()
                # Plays some samples
                playout_reward, results, rollout_path = self._forward_explore()
                # Train on collected samples
                self._optimize_path(playout_reward, results, rollout_path)
                if (self.id == 0) and (self.optimizer.get_global_step() % 100) == 0:
                    print(f'Global Step {self.optimizer.get_global_step()}')

                # Trigger save or other
                self.saver.after_optimization()                
                # pass
            # compare_models(self.resnet_model.resnet, self.resnet_network)
        except Exception as e:
            print(e)
            # TODO: add logging
            #self.logger.error(e.msg)
            raise e

    def stop(self):
        print("Stop initiated")
        self.exit.set()
