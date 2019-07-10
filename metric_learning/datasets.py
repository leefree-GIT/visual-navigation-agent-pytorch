

import json
from random import choice, shuffle

import h5py
import networkx as nx
import numpy as np
import torch
from networkx.readwrite import json_graph
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler


class BatchSampler(Sampler):

    def __init__(self, sampler, batch_size, drop_last):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for _, idx in enumerate(iter(self.sampler)):
            batch = idx
            yield batch

        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        return len(self.sampler) // self.batch_size


class TripletNavigationDataset(Dataset):
    def __init__(self, method, scene_name, target, mask_size, transform):

        # Create path to h5 file
        path = 'data/' + str(scene_name) + '.h5'

        # Load h5 file
        self.h5_file = h5py.File(path, 'r')

        # Load locations
        self.n_locations = len(self.h5_file['location'][()])

        # Load networkx graph
        self.networkx_graph = json_graph.node_link_graph(json.loads(
            self.h5_file["networkx_graph"][0]))

        # Compute shortest path between each states
        self.shortest_path = nx.shortest_path(self.networkx_graph)

        # Load shortest path distance between each states
        self.shortest_path_distance = self.h5_file['shortest_path_distance']

        # Transform used to project embedding
        self.transform = transform

        self.output_obs = False

    def set_output_obs(self, value):
        self.output_obs = value

    def _get_state(self, state_id):
        return self.h5_file['resnet_feature'][state_id][0]

    def __getitem__(self, index):
        # TARGET / ANCHOR
        target = self._get_state(index)
        # Use for embedding plot
        obs = []
        if self.output_obs:
            obs = self.h5_file["observation"][index]
            obs = self.transform(obs)

        candidate = self.shortest_path_distance[index][:]

        # POSITIVE
        # Get index
        candidate_positive_idx = np.transpose(np.nonzero(
            (candidate <= 2) & (candidate >= 1)))
        # Extract index
        candidate_positive_idx = [c[0] for c in candidate_positive_idx]

        # Fallback to only close
        if len(candidate_positive_idx) == 0:
            candidate_positive_idx = np.transpose(np.nonzero(
                (candidate <= 2) & (candidate >= 1)))
            # Extract index
            candidate_positive_idx = [c[0] for c in candidate_positive_idx]
            # Random choice
        positive_idx = choice(candidate_positive_idx)
        positive = self._get_state(positive_idx)

        # NEGATIVE
        # Negative is more than 6 step away
        candidate_negative_idx = np.transpose(
            np.nonzero((candidate >= 6)))

        # Random choice
        negative_idx = choice(candidate_negative_idx)[0]
        negative = self._get_state(negative_idx)

        # Hardest negative

        if np.isnan(target).any() or np.isinf(target).any():
            print("Target nan", index)
        if np.isnan(positive).any() or np.isinf(positive).any():
            print("Positive nan", positive_idx)
        if np.isnan(negative).any() or np.isinf(negative).any():
            print("Negative nan", negative_idx)
        return target, positive, negative, obs, index

    def __len__(self):
        return self.n_locations


class OnlineTripletNavigationDataset(Dataset):
    def __init__(self, method, scene_name, target, mask_size, transform):

        # Create path to h5 file
        path = 'data/' + str(scene_name) + '.h5'

        # Load h5 file
        self.h5_file = h5py.File(path, 'r')

        # Load locations
        self.n_locations = len(self.h5_file['location'][()])

        # Load networkx graph
        self.networkx_graph = json_graph.node_link_graph(json.loads(
            self.h5_file["networkx_graph"][0]))

        # Compute shortest path between each states
        self.shortest_path = nx.shortest_path(self.networkx_graph)

        # Load shortest path distance between each states
        self.shortest_path_distance = self.h5_file['shortest_path_distance']

        # Transform used to project embedding
        self.transform = transform

        self.output_obs = False

    def set_output_obs(self, value):
        self.output_obs = value

    def _get_state(self, state_id):
        return self.h5_file['resnet_feature'][state_id][0]

    def __getitem__(self, index):
        # TARGET / ANCHOR
        target = self._get_state(index)
        # Use for embedding plot
        obs = []
        if self.output_obs:
            obs = self.h5_file["observation"][index]
            obs = self.transform(obs)

        candidate = self.shortest_path_distance[index][:]

        # POSITIVE
        # Get index
        candidate_positive_idx = np.transpose(np.nonzero(
            (candidate <= 2) & (candidate >= 1)))
        # Extract index
        candidate_positive_idx = [c[0] for c in candidate_positive_idx]

        # Fallback to only close
        if len(candidate_positive_idx) == 0:
            candidate_positive_idx = np.transpose(np.nonzero(
                (candidate <= 2) & (candidate >= 1)))
            # Extract index
            candidate_positive_idx = [c[0] for c in candidate_positive_idx]

        # NEGATIVE
        # Negative is more than 6 step away
        candidate_negative_idx = np.transpose(
            np.nonzero((candidate >= 6)))
        candidate_negative_idx = [c[0] for c in candidate_negative_idx]

        out_positive_idx = torch.zeros(self.n_locations)
        out_negative_idx = torch.zeros(self.n_locations)

        out_positive_idx[candidate_positive_idx] = 1
        out_negative_idx[candidate_negative_idx] = 1
        # if np.isnan(target).any() or np.isinf(target).any():
        #     print("Target nan", index)
        # if np.isnan(positive).any() or np.isinf(positive).any():
        #     print("Positive nan", positive_idx)
        # if np.isnan(negative).any() or np.isinf(negative).any():
        #     print("Negative nan", negative_idx)
        return target, out_positive_idx, out_negative_idx, obs, index

    def __len__(self):
        return self.n_locations


class TripletMNIST(Dataset):
    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset
        self.train = self.mnist_dataset.train
        self.transform = self.mnist_dataset.transform

        self.train_labels = self.mnist_dataset.targets
        self.train_data = self.mnist_dataset.data
        self.labels_set = set(self.train_labels.numpy())
        self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                 for label in self.labels_set}

    def set_output_obs(self, value):
        return

    def __getitem__(self, index):
        img1, label1 = self.train_data[index], self.train_labels[index].item()
        positive_index = index
        while positive_index == index:
            positive_index = np.random.choice(self.label_to_indices[label1])
        negative_label = np.random.choice(
            list(self.labels_set - set([label1])))
        negative_index = np.random.choice(
            self.label_to_indices[negative_label])
        img2 = self.train_data[positive_index]
        img3 = self.train_data[negative_index]

        img1 = Image.fromarray(img1.numpy(), mode='L')
        img2 = Image.fromarray(img2.numpy(), mode='L')
        img3 = Image.fromarray(img3.numpy(), mode='L')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        return img1, img2, img3, [], index

    def __len__(self):
        return len(self.mnist_dataset)
