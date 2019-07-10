#!/usr/bin/env python
import argparse
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
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


def train_triplet(method, mask_size):
    # Create summary writer to keep track of loss and embedding
    writer = SummaryWriter("EXPERIMENTS/METRICS/logs/" +
                           datetime.now().strftime('%b%d_%H-%M-%S'))
    cuda = torch.cuda.is_available()
    device = torch.device("cpu")
    if cuda:
        device = torch.device("cuda")

    # Create transform for embedding plot
    obs_transform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize(64),
                                        transforms.ToTensor()
                                        ])
    # Load dataset

    mode = "Online"  # MNIST, Online, Offline
    if mode == "Offline":
        dataset = TripletNavigationDataset(
            method, "FloorPlan1", {"object": "Microwave"}, mask_size, obs_transform)
        # Load Triplet Network
        model = TripletNetwork(ResnetEmbeddingNet()).to(device)
    elif mode == "Online":
        dataset = OnlineTripletNavigationDataset(
            method, "FloorPlan1", {"object": "Microwave"}, mask_size, obs_transform)
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

    train_loader = DataLoader(dataset, batch_size=16,
                              shuffle=True, pin_memory=True, num_workers=1)

    # Load Triplet loss
    if mode != "Online":
        lossFun = TripletLoss(margin=1).to(device)
    else:
        lossFun = OnlineTripletLoss(margin=1).to(device)

    # Use blog optimizer/scheduler
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = lr_scheduler.StepLR(optimizer, 16, gamma=0.1, last_epoch=-1)

    # We will train the model (unfreeze)
    model.train()

    # Max train epoch
    MAX_EPOCH = 1000
    EMBEDDING_EPOCH = 10
    t_epoch = tqdm(range(MAX_EPOCH), desc="Epoch")
    for epoch in t_epoch:
        losses = []
        embedding = None
        obs_embedding = None
        indexes_batch = None
        scheduler.step()

        # Only output obs for embedding plot
        if mode == 'Offline' and epoch % EMBEDDING_EPOCH == 0:
            dataset.set_output_obs(True)
        elif mode == 'Offline':
            dataset.set_output_obs(False)

        for batch in tqdm(train_loader, desc="Batch"):
            optimizer.zero_grad()
            (targets, positives, negatives, observations,
             indexes) = batch

            # for enu, indx in enumerate(indexes):
            #     fig = plt.figure()
            #     fig.add_subplot(1, 3, 1)
            #     plt.imshow(dataset.h5_file['observation'][indexes[enu]])
            #     fig.add_subplot(1, 3, 2)
            #     plt.imshow(dataset.h5_file['observation'][negative_idxs[enu]])
            #     fig.add_subplot(1, 3, 3)
            #     plt.imshow(dataset.h5_file['observation'][positive_idxs[enu]])
            #     plt.show()
            if mode != "Online":
                targets = targets.to(device)
                positives = positives.to(device)
                negatives = negatives.to(device)
                emb_targets, emb_positive, emb_negative = model(
                    targets, positives, negatives)

                loss = lossFun(emb_targets, emb_positive, emb_negative)
                losses.append(loss.cpu().item())
                loss.backward()

                optimizer.step()

            elif mode == "Online":
                targets = targets.to(device)
                emb_targets = model.get_embedding(targets)
                loss = lossFun(emb_targets, indexes, positives, negatives)
                losses.append(loss.cpu().item())
                loss.backward()

                optimizer.step()
            # Store data for embedding plot
            if mode != "Online" and epoch % EMBEDDING_EPOCH == 0:
                if mode != "MNIST":
                    # Only plot front view (not up or down view)
                    for idx_enum, idx in enumerate(indexes):
                        ob = observations[idx_enum].detach().numpy()
                        if embedding is None:
                            embedding = emb_targets[idx_enum].unsqueeze(0)
                        else:
                            embedding = torch.cat(
                                [embedding, emb_targets[idx_enum].unsqueeze(0)], dim=0)

                        if obs_embedding is None:
                            obs_embedding = observations[idx_enum].unsqueeze(
                                0)
                        else:
                            obs_embedding = torch.cat(
                                [obs_embedding, observations[idx_enum].unsqueeze(0)], dim=0)
                else:
                    if embedding is None:
                        embedding = emb_targets
                    else:
                        embedding = torch.cat(
                            [embedding, emb_targets], dim=0)
                    if obs_embedding is None:
                        obs_embedding = targets
                    else:
                        obs_embedding = torch.cat(
                            [obs_embedding, targets], dim=0)

                    if indexes_batch is None:
                        indexes_batch = indexes
                    else:
                        indexes_batch = torch.cat(
                            [indexes_batch, indexes], dim=0)

        # Plot loss
        losses = np.array(losses)
        mean_loss = np.mean(losses[np.logical_not(np.isnan(losses))])
        t_epoch.set_postfix(loss=mean_loss)
        writer.add_scalar('loss', mean_loss, epoch)

        # Plot embedding
        if mode != "Online" and epoch % EMBEDDING_EPOCH == 0:
            embedding = embedding.cpu().detach()
            obs_embedding = obs_embedding.cpu().detach()
            if mode != "MNIST":
                writer.add_embedding(
                    mat=embedding, label_img=obs_embedding, global_step=epoch)
            else:
                indexes_batch = indexes_batch.cpu().detach()
                writer.add_embedding(
                    mat=embedding, label_img=obs_embedding, global_step=epoch)
                # plot_embeddings(embedding.numpy(),
                #                 dataset.train_labels[indexes_batch.numpy()])
                # plt.show()
            writer._get_file_writer().flush()
            del obs_embedding
            del embedding

    writer.close()
    torch.save(model.state_dict(),
               "EXPERIMENTS/METRICS/checkpoint" + str(MAX_EPOCH) + ".pth")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Metric learning for agent navigation")
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
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.set_num_threads(1)
    method = "word2vec"
    mask_size = 16
    train_triplet(method, mask_size)
