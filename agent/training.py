from agent.network import SharedNetwork, SceneSpecificNetwork, SharedResnet, compare_models
from agent.training_thread import TrainingThread
from agent.gpu_thread import GPUThread
from agent.summary_thread import SummaryThread
from agent.optim import SharedRMSprop
from typing import Collection, List
import torch.nn as nn
import torch.multiprocessing as mp 
import logging
import sys
import torch
import os
import threading
import h5py
from skimage.transform import resize
from torchvision import transforms
import torchvision.transforms.functional as F
import numpy as np
from contextlib import suppress
import re

from agent.resnet import resnet50

from tensorboardX import SummaryWriter
import logging
logging.basicConfig(level=logging.DEBUG)

import imp
MainModel = imp.load_source('MainModel', "agent/resnet/resnet50.py")

# Preprocess obs to match resnet input 

class TrainingSaver:
    def __init__(self, shared_network, scene_networks, optimizer, config):
        self.checkpoint_path = config.get('checkpoint_path', 'model/checkpoint-{checkpoint}.pth')
        self.saving_period = config.get('saving_period')
        self.shared_network = shared_network
        self.scene_networks = scene_networks
        self.optimizer = optimizer
        self.config = config

    def after_optimization(self):
        iteration = self.optimizer.get_global_step()
        if iteration % self.saving_period == 0:
            print('Saving training session')
            self.save()

    def save(self):
        iteration = self.optimizer.get_global_step()
        filename = self.checkpoint_path.replace('{checkpoint}', str(iteration))
        model = dict()
        model['navigation'] = self.shared_network.state_dict()
        for key, val in self.scene_networks.items():
            model[f'navigation/{key}'] = val.state_dict()
        model['optimizer'] = self.optimizer.state_dict()
        model['config'] = self.config
        
        with suppress(FileExistsError):
            os.makedirs(os.path.dirname(filename))
        torch.save(model, open(filename, 'wb'))

    def restore(self, state):
        if 'optimizer' in state and self.optimizer is not None: self.optimizer.load_state_dict(state['optimizer'])
        if 'config' in state: 
            conf = self.config
            self.config = state['config']
            for k, v in conf.items():
                self.config[k] = v

        self.shared_network.load_state_dict(state['navigation'])

        tasks = self.config.get('task_list')
        for scene in tasks.keys():
            self.scene_networks[scene].load_state_dict(state[f'navigation/{scene}'])

class TrainingOptimizer:
    def __init__(self, grad_norm, optimizer, scheduler):
        self.optimizer : torch.optim.Optimizer = optimizer
        self.scheduler = scheduler
        self.grad_norm = grad_norm
        self.global_step = torch.tensor(0)
        self.lock = mp.Lock()

    def state_dict(self):
        state_dict = dict()
        state_dict['optimizer'] = self.optimizer.state_dict()
        state_dict['scheduler'] = self.scheduler.state_dict()
        state_dict["global_step"] = self.global_step
        return state_dict

    def share_memory(self):
        self.global_step.share_memory_()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scheduler.load_state_dict(state_dict['scheduler'])
        self.global_step.copy_(state_dict['global_step'])
    
    def get_global_step(self):
        return self.global_step.item()

        
    def _ensure_shared_grads(self, local_params, shared_params):
        for param, shared_param in zip(local_params, shared_params):
            if shared_param.grad is not None:
                return
            shared_param._grad = param.grad

    def optimize(self, loss, local_params, shared_params):
        local_params = list(local_params)
        shared_params = list(shared_params)

        # Fix the optimizer property after unpickling
        self.scheduler.optimizer = self.optimizer
        self.scheduler.step(self.global_step.item())

        # Increment step
        with self.lock:
            self.global_step.copy_(torch.tensor(self.global_step.item() + 1))
            
        self.optimizer.zero_grad()

        # Calculate the new gradient with the respect to the local network
        loss.backward()

        # Clip gradient
        torch.nn.utils.clip_grad_norm_(local_params, self.grad_norm)
            
        self._ensure_shared_grads(local_params, shared_params)
        self.optimizer.step()
    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

class AnnealingLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, total_epochs, max_t, last_epoch=-1):
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.total_epochs = total_epochs
        self.max_t = max_t
        super(AnnealingLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * (1.0 - (self.last_epoch * self.max_t) / self.total_epochs)
                for base_lr in self.base_lrs]

class Training:
    def __init__(self, device, config):
        self.device = device
        self.config = config
        self.logger : logging.Logger = self._init_logger()
        self.learning_rate = config.get('learning_rate')
        self.rmsp_alpha = config.get('rmsp_alpha')
        self.rmsp_epsilon = config.get('rmsp_epsilon')
        self.grad_norm = config.get('grad_norm', 40.0)
        self.tasks = config.get('task_list')
        self.checkpoint_path = config.get('checkpoint_path', 'model/checkpoint-{checkpoint}.pth')
        self.max_t = config.get('max_t')
        self.num_thread = config.get('num_thread', 1)
        self.total_epochs = config.get('total_step')
        self.initialize()

    @staticmethod
    def load_checkpoint(config, fail = True):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint_path = config.get('checkpoint_path', 'model/checkpoint-{checkpoint}.pth')
        total_epochs = config.get('total_step')
        files = os.listdir(os.path.dirname(checkpoint_path))
        base_name = os.path.basename(checkpoint_path)
        
        # Find latest checkpoint
        # TODO: improve speed
        restore_point = None
        if base_name.find('{checkpoint}') != -1:
            regex = re.escape(base_name).replace(re.escape('{checkpoint}'), '(\d+)')
            points = [(fname, int(match.group(1))) for (fname, match) in ((fname, re.match(regex, fname),) for fname in files) if not match is None]
            if len(points) == 0:
                if fail:
                    raise Exception('Restore point not found')
                else: return None
            
            (base_name, restore_point) = max(points, key = lambda x: x[1])

            
        print(f'Restoring from checkpoint {restore_point}')
        state = torch.load(open(os.path.join(os.path.dirname(checkpoint_path), base_name), 'rb'))
        training = Training(device, state['config'] if 'config' in state else config)
        training.saver.restore(state)        
        return training

    def initialize(self):
        # Shared network
        self.shared_network = SharedNetwork()
        self.scene_networks = { key:SceneSpecificNetwork(4) for key in self.tasks.keys() }

        # Share memory
        self.shared_network.share_memory()
        for net in self.scene_networks.values():
            net.share_memory()

        # Callect all parameters from all networks
        parameters = list(self.shared_network.parameters())
        for net in self.scene_networks.values():
            parameters.extend(net.parameters())

        # Create optimizer
        optimizer = SharedRMSprop(parameters, eps=self.rmsp_epsilon, alpha=self.rmsp_alpha, lr=self.learning_rate)
        optimizer.share_memory()

        # Create scheduler
        scheduler = AnnealingLRScheduler(optimizer, self.total_epochs, self.max_t)

        # Create optimizer wrapper
        optimizer_wrapper = TrainingOptimizer(self.grad_norm, optimizer, scheduler)
        self.optimizer = optimizer_wrapper
        optimizer_wrapper.share_memory()

        # Initialize saver
        self.saver = TrainingSaver(self.shared_network, self.scene_networks, self.optimizer, self.config)
    
    def run(self):
        self.logger.info("Training started")
        self.print_parameters()

        # Prepare threads
        branches = []
        for scene in self.tasks.keys():
            it = 0
            for target in self.tasks.get(scene):
                target['id'] = it
                it = it + 1
                branches.append((scene, target))


        # If True use resnet to extract feature
        # If False use precomputed one
        use_resnet = self.config['use_resnet']
        print(f"Resnet {use_resnet}")


        def _createThread(id, task, i_queue, o_queue, evt, summary_queue, device):
            (scene, target) = task
            net = nn.Sequential(self.shared_network, self.scene_networks[scene])
            net.share_memory()

            if use_resnet:
                return TrainingThread(
                    id = id,
                    optimizer = self.optimizer,
                    network = net,
                    scene = scene,
                    saver = self.saver,
                    terminal_state = target,
                    device = self.device,
                    input_queue = i_queue,
                    output_queue = o_queue,
                    evt = evt,
                    summary_queue = summary_queue,
                    **self.config)
            else:
                return TrainingThread(
                    id = id,
                    optimizer = self.optimizer,
                    network = net,
                    scene = scene,
                    saver = self.saver,
                    terminal_state = target,
                    device = self.device,
                    input_queue = i_queue,
                    output_queue = o_queue,
                    evt = evt,
                    summary_queue = summary_queue,
                    **self.config)

        # Retrieve number of task
        num_scene_task =  len(branches)

        if self.num_thread < num_scene_task:
            self.num_thread = num_scene_task
            print('ERROR: num_thread must be higher than ', num_scene_task)

        self.threads = []


        # Queues will be used to pass feature to GPUThread, one to send, one to receive per Process
        input_queues = []
        output_queues = []
        summary_queue = mp.Queue()
        evt = mp.Event()
        for i in range(self.num_thread):
            input_queue = mp.Queue()
            output_queue = mp.Queue()
            input_queues.append(input_queue)
            output_queues.append(output_queue)

        # Create a summary thread to log
        self.summary = SummaryThread(self.config['log_path'], summary_queue)




        # Create GPUThread to handle feature computation
        if use_resnet:
            # Download pretrained resnet
            resnet_trained_pytorch = torch.load('agent/resnet/resnet50.pth')
            resnet_trained_pytorch.eval()
            resnet_custom = SharedResnet(resnet_trained_pytorch)
            h5_file_path = self.config.get('h5_file_path')
            self.gpu = GPUThread(resnet_custom, self.device, input_queues, output_queues, list(self.tasks.keys()), h5_file_path, evt)

        # Create at least 1 thread per task
        for i in range(self.num_thread):
            self.threads.append(_createThread(i, branches[i%num_scene_task], output_queues[i], input_queues[i], evt, summary_queue, self.device))
        

        
        # self.threads = [_createThread(i, task) for i, task in enumerate(branches)]
        print(f"Running for {self.total_epochs}")
        try:
            # Start the logger thread
            self.summary.start()

            # Start the gpu thread
            if use_resnet:
                self.gpu.start()

            # Then start the agents threads
            for thread in self.threads:
                thread.start()

            # Wait for agent
            for thread in self.threads:
                thread.join()
            
            # Wait for GPUThread
            if use_resnet:
                self.gpu.stop()
                self.gpu.join()

            # Wait for logger
            self.summary.stop()
            self.summary.join()

            # Save last checkpoint
            self.saver.save()
        except KeyboardInterrupt:
            # we will save the training
            print('Saving training session')
            self.saver.save()
            
            for thread in self.threads:
                thread.stop()
                thread.join()
            if use_resnet:
                self.gpu.stop()
                self.gpu.join()
            
            self.summary.stop()
            self.summary.join()
        

    def _init_logger(self):
        logger = logging.getLogger('agent')
        logger.setLevel(logging.INFO)
        logger.addHandler(logging.StreamHandler(sys.stdout))
        return logger

    def print_parameters(self):
        self.logger.info(f"- gamma: {self.config.get('gamma')}")
        self.logger.info(f"- learning rate: {self.config.get('learning_rate')}")
