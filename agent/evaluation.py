

import csv
import imp
import os
import random
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
from torch.nn import Sequential

from agent.environment.ai2thor_file import \
    THORDiscreteEnvironment as THORDiscreteEnvironmentFile
from agent.gpu_thread import GPUThread
from agent.method.aop import AOP
from agent.method.gcn import GCN
from agent.method.similarity_grid import SimilarityGrid
from agent.method.target_driven import TargetDriven
from agent.network import SceneSpecificNetwork, SharedNetwork
from agent.training import TrainingSaver
from agent.utils import find_restore_points, get_first_free_gpu
from torchvision import transforms


def prepare_csv(file, scene_task):
    f = open(file, 'w', newline='')
    writer = csv.writer(f)
    header = ['']
    header2lvl = ['Checkpoints']
    values = ['_reward', '_length', '_collision', '_success', '_spl']
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
        self.method = config['method']
        gpu_id = get_first_free_gpu(2000)
        self.device = torch.device("cuda:" + str(gpu_id))
        if self.method != "random":
            self.shared_net = SharedNetwork(
                self.config['method'], self.config.get('mask_size', 5)).to(self.device)
            self.scene_net = SceneSpecificNetwork(
                self.config['action_size']).to(self.device)

        self.checkpoints = []
        self.checkpoint_id = 0
        self.saver = None
        self.chk_numbers = None

    @staticmethod
    def load_checkpoints(config, fail=True):
        evaluation = Evaluation(config)
        checkpoint_path = config.get(
            'checkpoint_path', 'model/checkpoint-{checkpoint}.pth')

        checkpoints = []
        (base_name, chk_numbers) = find_restore_points(checkpoint_path, fail)
        if evaluation.method != "random":
            try:
                for chk_name in base_name:
                    state = torch.load(
                        open(os.path.join(os.path.dirname(checkpoint_path), chk_name), 'rb'))
                    checkpoints.append(state)
            except Exception as e:
                print("Error loading", e)
                exit()
        evaluation.saver = TrainingSaver(evaluation.shared_net,
                                         evaluation.scene_net, None, evaluation.config)
        evaluation.chk_numbers = chk_numbers
        evaluation.checkpoints = checkpoints
        return evaluation

    def restore(self):
        print('Restoring from checkpoint',
              self.chk_numbers[self.checkpoint_id])
        self.saver.restore(self.checkpoints[self.checkpoint_id])

    def next_checkpoint(self):
        self.checkpoint_id = (self.checkpoint_id + 1) % len(self.checkpoints)

    def run(self, show=False):
        self.method_class = None
        if self.method == 'word2vec' or self.method == 'word2vec_noconv' or self.method == 'word2vec_notarget' or self.method == 'word2vec_nosimi':
            self.method_class = SimilarityGrid(self.method)
        elif self.method == 'aop' or self.method == 'aop_we':
            self.method_class = AOP(self.method)
        elif self.method == 'target_driven':
            self.method_class = TargetDriven(self.method)
        elif self.method == 'gcn':
            self.method_class = GCN(self.method)

        # Init random seed
        random.seed(200)
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
            if self.method != "random":
                self.restore()
                self.next_checkpoint()
            for scene_scope, items in self.config['task_list'].items():
                if self.method != "random":
                    scene_net = self.scene_net
                    scene_net.eval()

                network = Sequential(self.shared_net, scene_net)
                network.eval()
                scene_stats[scene_scope] = dict()
                scene_stats[scene_scope]["length"] = list()
                scene_stats[scene_scope]["spl"] = list()
                scene_stats[scene_scope]["success"] = list()
                scene_stats[scene_scope]["spl_long"] = list()
                scene_stats[scene_scope]["success_long"] = list()

                for task_scope in items:

                    env = THORDiscreteEnvironmentFile(scene_name=scene_scope,
                                                      method=self.method,
                                                      reward=self.config['reward'],
                                                      h5_file_path=(lambda scene: self.config.get(
                                                          "h5_file_path").replace('{scene}', scene)),
                                                      terminal_state=task_scope,
                                                      action_size=self.config['action_size'],
                                                      mask_size=self.config.get(
                                                          'mask_size', 5),
                                                      bbox_method=self.config.get(
                                                          'bbox_method', None),
                                                      we_method=self.config.get('we_method', None))

                    ep_rewards = []
                    ep_lengths = []
                    ep_collisions = []
                    ep_actions = []
                    ep_start = []
                    ep_success = []
                    ep_spl = []
                    ep_shortest_distance = []
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
                            if self.method != "random":
                                policy, value, state = self.method_class.forward_policy(
                                    env, self.device, network)
                                with torch.no_grad():
                                    action = F.softmax(policy, dim=0).multinomial(
                                        1).data.cpu().numpy()[0]

                                if env.current_state_id not in state_ids:
                                    state_ids.append(env.current_state_id)
                            else:
                                action = np.random.randint(env.action_size)

                            env.step(action)
                            actions.append(action)
                            ep_reward += env.reward
                            terminal = env.terminal

                            if ep_t == 300:
                                break
                            if env.collided:
                                ep_collision += 1
                            ep_t += 1

                        ep_actions.append(actions)
                        ep_lengths.append(ep_t)
                        ep_rewards.append(ep_reward)
                        ep_shortest_distance.append(env.shortest_path_terminal(
                            ep_start[-1]))
                        ep_collisions.append(ep_collision)

                        # Compute SPL
                        spl = env.shortest_path_terminal(
                            ep_start[-1])/ep_t
                        ep_spl.append(spl)

                        if self.config['reward'] == 'soft_goal':
                            if env.success:
                                ep_success.append(True)
                            else:
                                ep_success.append(False)

                        elif ep_t < 300:
                            ep_success.append(True)
                        else:
                            ep_success.append(False)
                        log.write("episode #{} ends after {} steps, success : {}".format(
                            i_episode, ep_t, ep_success[-1]))

                    # Get indice of succeed episodes
                    ind_succeed_ep = [
                        i for (i, ep_suc) in enumerate(ep_success) if ep_suc]
                    ep_rewards = np.array(ep_rewards)
                    ep_lengths = np.array(ep_lengths)
                    ep_collisions = np.array(ep_collisions)
                    ep_spl = np.array(ep_spl)
                    ep_start = np.array(ep_start)

                    log.write('evaluation: %s %s' % (scene_scope, task_scope))
                    log.write('mean episode reward: %.2f' %
                              np.mean(ep_rewards[ind_succeed_ep]))
                    log.write('mean episode length: %.2f' %
                              np.mean(ep_lengths[ind_succeed_ep]))
                    log.write('mean episode collision: %.2f' %
                              np.mean(ep_collisions[ind_succeed_ep]))
                    ep_success_percent = (
                        (len(ind_succeed_ep) / self.config['num_episode']) * 100)
                    log.write('episode success: %.2f%%' %
                              ep_success_percent)

                    ep_spl_mean = np.sum(ep_spl[ind_succeed_ep]) / self.config['num_episode']
                    log.write('episode SPL: %.3f' % ep_spl_mean)

                    # Stat on long path
                    ind_succeed_far_start = []
                    ind_far_start = []
                    for i, short_dist in enumerate(ep_shortest_distance):
                        if short_dist > 5:
                            if ep_success[i]:
                                ind_succeed_far_start.append(i)
                            ind_far_start.append(i)

                    nb_long_episode = len(ind_far_start)
                    if nb_long_episode == 0:
                        nb_long_episode = 1
                    ep_success_long_percent = (
                        (len(ind_succeed_far_start) / nb_long_episode) * 100)
                    log.write('episode > 5 success: %.2f%%' %
                              ep_success_long_percent)
                    ep_spl_long_mean = np.sum(ep_spl[ind_succeed_far_start]) / nb_long_episode
                    log.write('episode SPL > 5: %.3f' % ep_spl_long_mean)
                    log.write('nb episode > 5: %d' % nb_long_episode)
                    log.write('')

                    scene_stats[scene_scope]["length"].extend(
                        ep_lengths[ind_succeed_ep])
                    scene_stats[scene_scope]["spl"].append(ep_spl_mean)
                    scene_stats[scene_scope]["success"].append(
                        ep_success_percent)
                    scene_stats[scene_scope]["spl_long"].append(
                        ep_spl_long_mean)
                    scene_stats[scene_scope]["success_long"].append(
                        ep_success_long_percent)

                    tmpData = [np.mean(
                        ep_rewards), np.mean(ep_lengths), np.mean(ep_collisions), ep_success_percent, ep_spl, ind_succeed_ep]
                    resultData = np.hstack((resultData, tmpData))

                    # Show best episode from evaluation
                    # We will print the best (lowest step), median, and worst
                    if show:
                        # Find episode based on episode length
                        sorted_ep_lengths = np.sort(ep_lengths[ind_succeed_ep])

                        # Best is the first episode in the sorted list but we want more than 10 step
                        index_best = 0
                        for idx, ep_len in enumerate(sorted_ep_lengths):
                            if ep_len >= 10:
                                index_best = idx
                                break
                        index_best = np.where(
                            ep_lengths[ind_succeed_ep] == sorted_ep_lengths[index_best])
                        index_best = index_best[0][0]

                        # Worst is the last episode in the sorted list
                        index_worst = np.where(
                            ep_lengths[ind_succeed_ep] == sorted_ep_lengths[-1])
                        index_worst = index_worst[0][0]

                        # Median is half the array size
                        index_median = np.where(
                            ep_lengths[ind_succeed_ep] == sorted_ep_lengths[len(sorted_ep_lengths)//2])
                        # Extract index
                        index_median = index_median[0][0]

                        names_video = ['best', 'median', 'worst']

                        # Create dir if not exisiting
                        directory = os.path.join(
                            self.config['base_path'], 'video', str(chk_id))
                        if not os.path.exists(directory):
                            os.makedirs(directory)
                        for idx_name, idx in enumerate([index_best, index_median, index_worst]):
                            # Create video to save
                            height, width, layers = np.shape(
                                env.observation)
                            video_name = os.path.join(directory, scene_scope + '_' +
                                                      task_scope['object'] + '_' +
                                                      names_video[idx_name] + '_' +
                                                      str(ep_lengths[idx]) + '.avi')
                            FPS = 5
                            video = cv2.VideoWriter(
                                video_name, cv2.VideoWriter_fourcc(*"MJPG"), FPS, (width, height))
                            # Retrieve start position
                            state_id_best = ep_start[ind_succeed_ep][idx]
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
                    if False and self.method != "random" and self.method != "gcn":
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
                log.write('%s: %.2f steps | %.3f spl | %.2f%% success | %.3f spl > 5 | %.2f%% success > 5' %
                          (scene_scope, np.mean(scene_stats[scene_scope]["length"]), np.mean(
                              scene_stats[scene_scope]["spl"]), np.mean(
                              scene_stats[scene_scope]["success"]),
                              np.mean(
                              scene_stats[scene_scope]["spl_long"]),
                              np.mean(
                              scene_stats[scene_scope]["success_long"])))
            # Write data to csv
            writer_csv.writerow(list(resultData))
            break


'''
# Load weights trained on tensorflow
data = pickle.load(
    open(os.path.join(__file__, '..\\..\\weights.p'), 'rb'), encoding='latin1')
def convertToStateDict(data):
    return {key:torch.Tensor(v) for (key, v) in data.items()}

shared_net.load_state_dict(convertToStateDict(data['navigation']))
for key in TASK_LIST.keys():
    scene_nets[key].load_state_dict(convertToStateDict(data[f'navigation/{key}']))'''
