

from agent.network import SharedNetwork, SceneSpecificNetwork, SharedResnet
from agent.resnet import resnet50
from agent.environment import THORDiscreteEnvironment
from agent.training import TrainingSaver
from agent.utils import find_restore_point
import torch.nn.functional as F
import torch
import pickle
import os
import numpy as np
import re
from itertools import groupby

import torch.multiprocessing as mp


from agent.gpu_thread import GPUThread

from agent.constants import ACTION_SPACE_SIZE
from agent.constants import VERBOSE
import time

import imp
MainModel = imp.load_source('MainModel', "agent/resnet/resnet50.py")


def export_to_csv(data, file):
    import csv
    with open(file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        for k, g in groupby(sorted(data, key = lambda x: x[0]), key = lambda x: x[0]):
            g = list(g)
            header = [k, '']
            header.extend((np.mean(a) for a in list(zip(*g))[2:]))
            writer.writerow(header)
            for item in g:
                writer.writerow(list(item))
    print(f'CSV file stored "{file}"')


class Evaluation:
    def __init__(self, config):
        self.config = config
        self.shared_net = SharedNetwork()
        self.scene_nets = { key:SceneSpecificNetwork(ACTION_SPACE_SIZE) for key in config['task_list'].keys() }

    @staticmethod
    def load_checkpoint(config, fail = True):
        checkpoint_path = config.get('checkpoint_path', 'model/checkpoint-{checkpoint}.pth')
        
        import os
        (base_name, restore_point) = find_restore_point(checkpoint_path, fail)
        print(f'Restoring from checkpoint {os.path.basename(checkpoint_path)}')
        try:
            state = torch.load(open(os.path.join(os.path.dirname(checkpoint_path), base_name), 'rb'))
        except:
            print("Error loading")
            exit()
        evaluation = Evaluation(config)
        saver = TrainingSaver(evaluation.shared_net, evaluation.scene_nets, None, evaluation.config)
        saver.restore(state)        
        return evaluation
        
    
    def run(self):
        scene_stats = dict()
        resultData = []
        use_resnet = self.config.get("resnet")
        if use_resnet:
            mp.set_start_method('spawn')
            device = torch.device("cuda")
            # Download pretrained resnet
            resnet_trained_pytorch = torch.load('agent/resnet/resnet50.pth')
            resnet_custom = SharedResnet(resnet_trained_pytorch)
            resnet_custom.to(device)
            resnet_custom.share_memory()

            input_queue = mp.Queue()
            output_queue = mp.Queue()
            h5_file_path = self.config.get('h5_file_path')
            evt = mp.Event()
            gpu_thread = GPUThread(resnet_custom, device, [input_queue], [output_queue], list(self.config['task_list'].keys()), h5_file_path, evt)
            gpu_thread.start()
        for scene_scope, items in self.config['task_list'].items():
            scene_net = self.scene_nets[scene_scope]
            scene_stats[scene_scope] = list()
            for task_scope in items:
                if use_resnet:
                    env = THORDiscreteEnvironment(
                        scene_name=scene_scope,
                        input_queue=output_queue,
                        output_queue=input_queue,
                        evt = evt,
                        use_resnet=use_resnet,
                        h5_file_path=(lambda scene: self.config.get("h5_file_path", "D:\\datasets\\visual_navigation_precomputed\\{scene}.h5").replace('{scene}', scene)),
                        terminal_state_id=int(task_scope)
                    )
                else:
                    env = THORDiscreteEnvironment(
                    scene_name=scene_scope,
                    use_resnet=use_resnet,
                    h5_file_path=(lambda scene: self.config.get("h5_file_path", "D:\\datasets\\visual_navigation_precomputed\\{scene}.h5").replace('{scene}', scene)),
                    terminal_state = task_scope
                )

                ep_rewards = []
                ep_lengths = []
                ep_collisions = []
                for i_episode in range(self.config['num_episode']):
                    env.reset()
                    terminal = False
                    ep_reward = 0
                    ep_collision = 0
                    ep_t = 0
                    while not terminal:
                        state = torch.Tensor(env.render(mode='resnet_features'))
                        target = torch.Tensor(env.render_target(mode='resnet_features'))
                        (policy, value,) = scene_net.forward(self.shared_net.forward((state, target,)))

                        with torch.no_grad():
                            action = F.softmax(policy, dim=0).multinomial(1).data.numpy()[0]

                        env.step(action)
                        terminal = env.terminal

                        if ep_t == 5000: break
                        if env.collided: ep_collision += 1
                        ep_reward += env.reward
                        ep_t += 1

                    ep_lengths.append(ep_t)
                    ep_rewards.append(ep_reward)
                    ep_collisions.append(ep_collision)
                    if VERBOSE: print("episode #{} ends after {} steps".format(i_episode, ep_t))

                
                print('evaluation: %s %s' % (scene_scope, task_scope))
                print('mean episode reward: %.2f' % np.mean(ep_rewards))
                print('mean episode length: %.2f' % np.mean(ep_lengths))
                print('mean episode collision: %.2f' % np.mean(ep_collisions))
                scene_stats[scene_scope].extend(ep_lengths)
                resultData.append((scene_scope, str(task_scope), np.mean(ep_rewards), np.mean(ep_lengths), np.mean(ep_collisions),))

        print('\nResults (average trajectory length):')
        for scene_scope in scene_stats:
            print('%s: %.2f steps'%(scene_scope, np.mean(scene_stats[scene_scope])))
        
        if 'csv_file' in self.config and self.config['csv_file'] is not None:
            export_to_csv(resultData, self.config['csv_file'])
        if use_resnet:
            gpu_thread.stop()
            gpu_thread.join()

'''
# Load weights trained on tensorflow
data = pickle.load(open(os.path.join(__file__, '..\\..\\weights.p'), 'rb'), encoding='latin1')
def convertToStateDict(data):
    return {key:torch.Tensor(v) for (key, v) in data.items()}

shared_net.load_state_dict(convertToStateDict(data['navigation']))
for key in TASK_LIST.keys():
    scene_nets[key].load_state_dict(convertToStateDict(data[f'navigation/{key}']))'''
