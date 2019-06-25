#!/usr/bin/env python
import argparse
import multiprocessing as mp
import sys
from itertools import groupby

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from agent.environment.ai2thor_file import \
    THORDiscreteEnvironment as THORDiscreteEnvironmentFile
from agent.environment.ai2thor_real import \
    THORDiscreteEnvironment as THORDiscreteEnvironmentReal
from agent.evaluation import Evaluation
from agent.gpu_thread import GPUThread
from agent.network import SceneSpecificNetwork, SharedNetwork, SharedResnet
from agent.training import TrainingSaver
from agent.utils import find_restore_points, populate_config


class EvalEmbeddings:
    def __init__(self, config):
        self.config = config
        self.shared_net = SharedNetwork()
        self.scene_nets = {key: SceneSpecificNetwork(
            self.config['action_size']) for key in config['task_list'].keys()}
        self.checkpoints = []
        self.checkpoint_id = 0
        self.saver = None
        self.chk_numbers = None

    @staticmethod
    def load_checkpoints(config, fail=True):
        checkpoint_path = config.get(
            'checkpoint_path', 'model/checkpoint-{checkpoint}.pth')

        import os
        checkpoints = []
        (base_name, chk_numbers) = find_restore_points(checkpoint_path, fail)
        try:
            for chk_name in base_name:
                state = torch.load(
                    open(os.path.join(os.path.dirname(checkpoint_path), chk_name), 'rb'))
                checkpoints.append(state)
        except Exception as e:
            print("Error loading", e)
            exit()
        evalembed = EvalEmbeddings(config)
        evalembed.chk_numbers = chk_numbers
        evalembed.checkpoints = checkpoints
        evalembed.saver = TrainingSaver(evalembed.shared_net,
                                        evalembed.scene_nets, None, evalembed.config)
        return evalembed

    def restore(self):
        print('Loading from checkpoint',
              self.chk_numbers[self.checkpoint_id])
        self.saver.restore(self.checkpoints[self.checkpoint_id])

    def next_checkpoint(self):
        self.checkpoint_id = (self.checkpoint_id + 1) % len(self.checkpoints)

    def run(self):
        scene_stats = dict()
        resultData = []
        writer = SummaryWriter(self.config['log_path'])

        for chk in range(len(self.checkpoints)):
            chk_name = str(self.chk_numbers[self.checkpoint_id])
            for scene_scope, items in self.config['task_list'].items():
                self.shared_net.eval()
                for task_scope in items:
                    embedding_vectors = []
                    env = THORDiscreteEnvironmentFile(
                        scene_name=scene_scope,
                        h5_file_path=(lambda scene: self.config.get(
                            "h5_file_path", "D:\\datasets\\visual_navigation_precomputed\\{scene}.h5").replace('{scene}', scene)),
                        terminal_state=task_scope,
                        action_size=self.config['action_size']
                    )
                    env.reset()
                    for state in trange(env.n_locations):
                        env.current_state_id = state
                        state = torch.Tensor(
                            env.render(mode='resnet_features'))
                        target = torch.Tensor(
                            env.render_target(mode='resnet_features'))
                        object_mask = torch.Tensor(env.render_mask())

                        embedding_vector = self.shared_net.forward(
                            (state, target, object_mask,))

                        embedding_vectors.append(
                            embedding_vector.detach().numpy())

                    print(len(embedding_vectors))
                    obs = env.h5_file['observation']
                    obs = np.transpose(obs, (0, 3, 1, 2))
                    obs = torch.from_numpy(obs)
                    print(obs.size())

                    writer.add_embedding(
                        embedding_vectors, label_img=obs,
                        tag=chk_name + '_' + scene_scope + '_' + task_scope["object"])
            break
        writer.close()


if __name__ == '__main__':
    argparse.ArgumentParser(description="")
    parser = argparse.ArgumentParser(description='Deep reactive agent.')
    parser.add_argument('--h5_file_path', type=str,
                        default='/app/data/{scene}_keras.h5')
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--csv_file', type=str, default=None)
    parser.add_argument('--log_arg', type=int, default=0)

    # Use experiment.json
    parser.add_argument('--exp', '-e', type=str,
                        help='Experiment parameters.json file', required=True)

    args = vars(parser.parse_args())
    if args['checkpoint_path'] is not None:
        args = populate_config(args, mode='eval', checkpoint=False)
    else:
        args = populate_config(args, mode='eval')

    t = EvalEmbeddings.load_checkpoints(args)
    t.run()
