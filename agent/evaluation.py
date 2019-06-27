

import csv
import imp
import sys
from itertools import groupby

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from PIL import Image
from tensorboardX import SummaryWriter

from agent.environment.ai2thor_file import \
    THORDiscreteEnvironment as THORDiscreteEnvironmentFile
from agent.environment.ai2thor_real import \
    THORDiscreteEnvironment as THORDiscreteEnvironmentReal
from agent.gpu_thread import GPUThread
from agent.network import SceneSpecificNetwork, SharedNetwork, SharedResnet
from agent.training import TrainingSaver
from agent.utils import find_restore_points

MainModel = imp.load_source('MainModel', "agent/resnet/resnet50.py")


def prepare_csv(file, scene_task):
    f = open(file, 'w', newline='')
    writer = csv.writer(f)
    header = ['']
    header2lvl = ['Checkpoints']
    values = ['_reward', '_length', '_collision', '_success']
    for scene_scope, tasks_scope in scene_task:
        for task in tasks_scope:
            for val in values:
                header.append(scene_scope)
                header2lvl.append(task['object'] + val)
    writer.writerow(header)
    writer.writerow(header2lvl)
    return writer


class Logger(object):
    def __init__(self, path="logfile.log"):
        self.terminal = sys.stdout
        self.log = open(path, "w")

    def write(self, message, term='\n'):
        self.terminal.write(message + term)
        self.log.write(message + term)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass

    def __del__(self):
        self.log.close()


class Evaluation:
    def __init__(self, config):
        self.config = config
        self.shared_net = SharedNetwork(self.config.get('mask_size', 5))
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
        evaluation = Evaluation(config)
        evaluation.chk_numbers = chk_numbers
        evaluation.checkpoints = checkpoints
        evaluation.saver = TrainingSaver(evaluation.shared_net,
                                         evaluation.scene_nets, None, evaluation.config)
        return evaluation

    def restore(self):
        print('Restoring from checkpoint',
              self.chk_numbers[self.checkpoint_id])
        self.saver.restore(self.checkpoints[self.checkpoint_id])

    def next_checkpoint(self):
        self.checkpoint_id = (self.checkpoint_id + 1) % len(self.checkpoints)

    def run(self, show=False):
        use_resnet = self.config.get("use_resnet")
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
            gpu_thread = GPUThread(resnet_custom, device, [input_queue], [
                                   output_queue], list(self.config['task_list'].keys()), h5_file_path, evt)
            gpu_thread.start()

        # Create csv writer with correct header
        if self.config['train']:
            writer_csv = prepare_csv(
                self.config['base_path'] + 'train.csv', self.config['task_list'].items())
        else:
            writer_csv = prepare_csv(
                self.config['base_path'] + 'eval.csv', self.config['task_list'].items())

        for chk_id in self.chk_numbers:
            resultData = [chk_id]
            scene_stats = dict()
            if self.config['train']:
                log = Logger(self.config['base_path'] + 'train' +
                             str(chk_id) + '.log')
            else:
                log = Logger(self.config['base_path'] + 'eval' +
                             str(chk_id) + '.log')
            self.restore()
            self.next_checkpoint()
            for scene_scope, items in self.config['task_list'].items():
                scene_net = self.scene_nets[scene_scope]
                scene_net.eval()
                scene_stats[scene_scope] = list()

                for task_scope in items:
                    if use_resnet:
                        env = THORDiscreteEnvironmentReal(
                            scene_name=scene_scope,
                            input_queue=output_queue,
                            output_queue=input_queue,
                            evt=evt,
                            use_resnet=use_resnet,
                            h5_file_path=(lambda scene: self.config.get(
                                "h5_file_path", "D:\\datasets\\visual_navigation_precomputed\\{scene}.h5").replace('{scene}', scene)),
                            terminal_state_id=int(task_scope)
                        )
                    else:
                        env = THORDiscreteEnvironmentFile(
                            scene_name=scene_scope,
                            use_resnet=use_resnet,
                            h5_file_path=(lambda scene: self.config.get(
                                "h5_file_path", "D:\\datasets\\visual_navigation_precomputed\\{scene}.h5").replace('{scene}', scene)),
                            terminal_state=task_scope,
                            action_size=self.config['action_size'],
                            mask_size=self.config.get('mask_size', 5)
                        )

                    ep_rewards = []
                    ep_lengths = []
                    ep_collisions = []
                    ep_actions = []
                    ep_start = []
                    ep_success = 0
                    embedding_vectors = []
                    state_ids = list()
                    for i_episode in range(self.config['num_episode']):
                        env.reset()
                        terminal = False
                        ep_reward = 0
                        ep_collision = 0
                        ep_t = 0
                        actions = []
                        ep_start.append(env.current_state_id)
                        while not terminal:
                            state = torch.Tensor(
                                env.render(mode='resnet_features'))
                            target = torch.Tensor(
                                env.render_target(mode='resnet_features'))
                            object_mask = torch.Tensor(env.render_mask())

                            embedding = self.shared_net.forward(
                                (state, target, object_mask,))
                            (policy, _,) = scene_net.forward(embedding)
                            embedding = embedding.detach().numpy()

                            with torch.no_grad():
                                action = F.softmax(policy, dim=0).multinomial(
                                    1).data.numpy()[0]

                            if env.current_state_id not in state_ids:
                                state_ids.append(env.current_state_id)
                                embedding_vectors.append(embedding)
                            else:
                                state_idx = state_ids.index(
                                    env.current_state_id)
                                embedding_vectors[state_idx] = (
                                    embedding_vectors[state_idx] + embedding)/2

                            env.step(action)
                            actions.append(action)
                            terminal = env.terminal

                            if ep_t == 5000:
                                break
                            if env.collided:
                                ep_collision += 1
                            ep_reward += env.reward
                            ep_t += 1

                        ep_actions.append(actions)
                        ep_lengths.append(ep_t)
                        ep_rewards.append(ep_reward)
                        if ep_t <= 500:
                            ep_success = ep_success + 1
                        ep_collisions.append(ep_collision)
                        log.write("episode #{} ends after {} steps".format(
                            i_episode, ep_t))

                    log.write('evaluation: %s %s' % (scene_scope, task_scope))
                    log.write('mean episode reward: %.2f' %
                              np.mean(ep_rewards))
                    log.write('mean episode length: %.2f' %
                              np.mean(ep_lengths))
                    log.write('mean episode collision: %.2f' %
                              np.mean(ep_collisions))
                    ep_success_percent = (
                        (ep_success / self.config['num_episode']) * 100)
                    log.write('episode success: %.2f%%' %
                              ep_success_percent)
                    log.write('')
                    scene_stats[scene_scope].extend(ep_lengths)

                    tmpData = [np.mean(
                        ep_rewards), np.mean(ep_lengths), np.mean(ep_collisions), ep_success_percent]
                    resultData = np.hstack((resultData, tmpData))

                    # Show best episode from evaluation
                    # We will print the best (lowest step), median, and worst
                    if show:
                        # Find episode based on episode length
                        sorted_ep_lengths = np.sort(ep_lengths)

                        # Best is the first episode in the sorted list but we want more than 10 step
                        index_best = 0
                        for idx, ep_len in enumerate(sorted_ep_lengths):
                            if ep_len >= 10:
                                index_best = idx
                                break
                        index_best = np.where(
                            ep_lengths == sorted_ep_lengths[index_best])
                        index_best = index_best[0][0]

                        # Worst is the last episode in the sorted list
                        index_worst = np.where(
                            ep_lengths == sorted_ep_lengths[-1])
                        index_worst = index_worst[0][0]

                        # Median is half the array size
                        index_median = np.where(
                            ep_lengths == sorted_ep_lengths[len(sorted_ep_lengths)//2])
                        # Extract index
                        index_median = index_median[0][0]

                        names_video = ['best', 'median', 'worst']

                        for idx_name, idx in enumerate([index_best, index_median, index_worst]):
                            # Create video to save
                            height, width, layers = np.shape(
                                env.observation)
                            video_name = self.config['base_path'] + scene_scope + '_' + \
                                task_scope['object'] + '_' + \
                                names_video[idx_name] + '_' + \
                                str(ep_lengths[idx]) + '.avi'
                            FPS = 5
                            video = cv2.VideoWriter(
                                video_name, cv2.VideoWriter_fourcc(*"MJPG"), FPS, (width, height))
                            # Retrieve start position
                            state_id_best = ep_start[idx]
                            env.reset()

                            # Set start position
                            env.current_state_id = state_id_best
                            for a in ep_actions[idx]:
                                img = cv2.cvtColor(
                                    env.observation, cv2.COLOR_BGR2RGB)
                                video.write(img)
                                env.step(a)
                            img = cv2.cvtColor(
                                env.observation, cv2.COLOR_BGR2RGB)
                            for i in range(10):
                                video.write(img)
                            video.release()

                    # Use tensorboard to plot embeddings
                    if self.config['train']:
                        embedding_writer = SummaryWriter(
                            self.config['log_path'] + '/embeddings_train/' + scene_scope + '_' + str(chk_id))
                    else:
                        embedding_writer = SummaryWriter(
                            self.config['log_path'] + '/embeddings_eval/' + scene_scope + '_' + str(chk_id))
                    obss = []

                    for indx, obs in enumerate(env.h5_file['observation']):
                        if indx in state_ids:
                            img = Image.fromarray(obs)
                            img = img.resize((64, 64))
                            obss.append(np.array(img))

                    obss = np.transpose(obss, (0, 3, 1, 2))
                    obss = obss / 255
                    obss = torch.from_numpy(obss)

                    # Write embeddings
                    embedding_writer.add_embedding(
                        embedding_vectors, label_img=obss,
                        tag=task_scope['object'], global_step=chk_id)

            log.write('\nResults (average trajectory length):')
            for scene_scope in scene_stats:
                log.write('%s: %.2f steps' %
                          (scene_scope, np.mean(scene_stats[scene_scope])))
            # Write data to csv
            writer_csv.writerow(list(resultData))
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
