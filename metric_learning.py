#!/usr/bin/env python
import argparse
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing
import torch.optim as optim
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from agent.environment.ai2thor_file import \
    THORDiscreteEnvironment as THORDiscreteEnvironmentFile
from agent.utils import populate_config
from metric_learning.datasets import *
from metric_learning.network import (EmbeddingNet, OnlineTripletLoss,
                                     ResnetEmbeddingNet, TripletLoss,
                                     TripletNetwork)
from torchvision import transforms
from torchvision.datasets import MNIST

mnist_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
          '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
          '#bcbd22', '#17becf']


def plot_embeddings(embeddings, targets, xlim=None, ylim=None):
    plt.figure(figsize=(10, 10))
    for i in range(10):
        inds = np.where(targets == i)[0]
        plt.scatter(embeddings[inds, 0],
                    embeddings[inds, 1], alpha=0.5, color=colors[i])
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(mnist_classes)


class MetricLearning():
    def __init__(self, device, scene, method, mask_size, mode, log_path):
        # Create transform for embedding plot
        obs_transform = transforms.Compose([transforms.ToPILImage(),
                                            transforms.Resize(64),
                                            transforms.ToTensor()
                                            ])
        # Load dataset
        if mode == "Offline":
            dataset = TripletNavigationDataset(
                method, scene, {"object": "Microwave"}, mask_size, obs_transform)
            # Load Triplet Network
            model = TripletNetwork(ResnetEmbeddingNet()).to(device)
        elif mode == "Online":
            dataset = OnlineTripletNavigationDataset(
                method, scene, {"object": "Microwave"}, mask_size, obs_transform)
            # dataset2 = OnlineTripletNavigationDataset(
            #     method, "FloorPlan2", {"object": "Microwave"}, mask_size, obs_transform)
            # dataset3 = OnlineTripletNavigationDataset(
            #     method, "FloorPlan2", {"object": "Microwave"}, mask_size, obs_transform)
            # dataset = ConcatDataset((dataset, dataset2, dataset3))
            # Load Triplet Network
            model = TripletNetwork(ResnetEmbeddingNet()).to(device)

        else:
            mean, std = 0.1307, 0.3081

            train_dataset = MNIST('data/MNIST', train=True, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((mean,), (std,))
                                  ]))
            dataset = TripletMNIST(train_dataset)
            # Load Triplet Network
            model = TripletNetwork(EmbeddingNet()).to(device)

        train_loader = DataLoader(dataset, batch_size=64,
                                  shuffle=True, pin_memory=True, num_workers=1)

        # Load Triplet loss
        if mode != "Online":
            lossFun = TripletLoss(margin=1).to(device)
        else:
            lossFun = OnlineTripletLoss(margin=1).to(device)

        self.model = model
        self.train_loader = train_loader
        self.dataset = dataset
        self.lossFun = lossFun
        self.writer_eval = None
        self.log_path = log_path
        self.mode = mode
        self.device = device

    def train_triplet(self):
        # Create summary writer to keep track of loss and embedding
        writer = SummaryWriter(self.log_path.replace("{folder}", ""))

        # Use blog optimizer/scheduler
        optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        scheduler = lr_scheduler.StepLR(
            optimizer, 16, gamma=0.1, last_epoch=-1)

        # We will train the model (unfreeze)
        self.model.train()

        # Max train epoch
        MAX_EPOCH = 100
        EMBEDDING_EPOCH = 10
        t_epoch = tqdm(range(MAX_EPOCH), desc="Epoch")
        for epoch in t_epoch:
            losses = []
            embedding = None
            obs_embedding = None
            indexes_batch = None
            scheduler.step()

            # Only output obs for embedding plot
            if epoch % EMBEDDING_EPOCH == 0:
                self.dataset.set_output_obs(True)
            else:
                self.dataset.set_output_obs(False)
            t_batch = tqdm(self.train_loader, desc="Batch")
            for batch in t_batch:
                optimizer.zero_grad()
                (targets, positives, negatives, observations,
                 indexes) = batch

                if self.mode != "Online":
                    targets = targets.to(self.device)
                    positives = positives.to(self.device)
                    negatives = negatives.to(self.device)
                    emb_targets, emb_positive, emb_negative = self.model(
                        targets, positives, negatives)

                    loss = self.lossFun(
                        emb_targets, emb_positive, emb_negative)
                    losses.append(loss.cpu().item())
                    loss.backward()

                    optimizer.step()

                elif self.mode == "Online":
                    targets = targets.to(self.device)
                    emb_targets = self.model.get_embedding(targets)
                    loss = self.lossFun(
                        emb_targets, indexes, positives, negatives)
                    losses.append(loss.cpu().item())
                    t_batch.set_postfix(loss=losses[-1])
                    loss.backward()

                    optimizer.step()
            # Store data for embedding plot
            if epoch % EMBEDDING_EPOCH == 0:
                self.eval_triplet(epoch)

            # Plot loss
            losses = np.array(losses)
            mean_loss = np.mean(losses[np.logical_not(np.isnan(losses))])
            t_epoch.set_postfix(loss=mean_loss)
            writer.add_scalar('loss', mean_loss, epoch)

        writer.close()
        torch.save(self.model.state_dict(),
                   self.log_path.replace("{folder}", "/embedding") + "checkpoint" + str(MAX_EPOCH) + ".pth")

    def eval_triplet(self, epoch=0):
        if self.writer_eval is None:
            self.writer_eval = SummaryWriter(
                self.log_path.replace("{folder}", "/embedding"))
        self.model.eval()
        self.dataset.set_output_obs(True)

        device = self.device
        model = self.model.to(device)

        # Plot embedding
        embeddings = np.zeros((len(self.train_loader.dataset), 2))
        if self.mode == "MNIST":
            obs_embeddings = np.zeros((
                len(self.train_loader.dataset)))
        else:
            obs_embeddings = np.zeros((
                len(self.train_loader.dataset), 3, 64, 85))
        indexes_embeddings = np.zeros((len(self.train_loader.dataset)))
        k = 0
        with torch.no_grad():
            for batch in tqdm(self.train_loader, desc="Batch eval"):
                # Get current batch
                (targets, positives, negatives, observations, indexes) = batch

                targets = targets.to(device)
                positives = positives.to(device)
                negatives = negatives.to(device)

                if self.mode != "Online":
                    emb_targets, emb_positive, emb_negative = model(
                        targets, positives, negatives)
                    embedding = emb_targets.cpu().detach()
                    obs_embedding = observations.cpu().detach()
                else:
                    emb_targets = model.get_embedding(targets)
                    embedding = emb_targets.cpu().detach()
                    obs_embedding = observations.cpu().detach()

                len_b = len(targets)
                embeddings[k: k+len_b, ...] = embedding.numpy()

                obs_embeddings[k: k +
                               len_b, ...] = obs_embedding.numpy()
                indexes_embeddings[k: k +
                                   len_b] = indexes.detach().cpu().numpy()

                k = k+len_b
        self.model.train()
        fig, ax = plt.subplots(figsize=(28.0, 25.0))
        if self.mode != "MNIST":
            x = []
            y = []
            obs_embeddings_front = []
            for idx, index in enumerate(indexes_embeddings):
                if self.dataset.rotations[int(index)][2] == 0.0:
                    x.append(embeddings[idx, 0])
                    y.append(embeddings[idx, 1])
                    obs_embeddings_front.append(obs_embeddings[idx])
            ax.scatter(x, y)
            for x0, y0, obs in zip(x, y, obs_embeddings_front):
                obs = np.transpose(obs, (1, 2, 0))
                obs = OffsetImage(obs, zoom=0.8)
                ab = AnnotationBbox(obs, (x0, y0), frameon=False)
                ax.add_artist(ab)

            labels = [str(x0) + '|' + str(y0)
                      for x0, y0 in zip(embeddings[:, 0], embeddings[:, 1])]

            self.writer_eval.add_embedding(
                mat=embeddings, metadata=labels, label_img=torch.from_numpy(obs_embeddings), global_step=epoch)
        else:
            plot_embeddings(embeddings, obs_embeddings)

        plt.savefig(self.log_path.replace(
            "{folder}", "/embedding") + "/fig_" + str(epoch))
        plt.close()


def main():

    parser = argparse.ArgumentParser(
        description="Metric learning for agent navigation")

    parser.add_argument('--eval', default=None, type=str,
                        help='eval production')
    args = vars(parser.parse_args())

    # # Use experiment.json
    # parser.add_argument('--exp', '-e', type=str,
    #                     help='Experiment parameters.json file', required=True)

    # args = vars(parser.parse_args())
    # args = populate_config(args)

    # if args.get('method', None) is None:
    #     print('ERROR Please choose a method in json file')
    #     print('- "aop"')
    #     print('- "word2vec"')
    #     print('- "target_driven"')

    #     exit()
    # else:
    #     method = args.get('method')
    #     if "aop" not in method or "word2vec" not in method or "target_driven" not in method:
    #         exit()

    # torch.manual_seed(args['seed'])
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.set_num_threads(1)
    method = "word2vec"
    mask_size = 16
    mode = "Online"  # MNIST, Online, Offline
    scene = "FloorPlan1"

    cuda = torch.cuda.is_available()
    device = torch.device("cpu")
    if cuda:
        device = torch.device("cuda")

    log_path = "EXPERIMENTS/METRICS/logs/" + \
        datetime.now().strftime('%b%d_%H-%M-%S') + "{folder}"
    metric = MetricLearning(
        device, scene, method, mask_size, mode, log_path)

    if args["eval"] is None:
        metric.train_triplet()
    else:
        metric.model.load_state_dict(torch.load(args["eval"]))
        metric.eval_triplet()


if __name__ == '__main__':
    main()
