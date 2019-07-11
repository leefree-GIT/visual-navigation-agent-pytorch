import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(1, 32, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2))

        self.fc = nn.Sequential(nn.Linear(64 * 4 * 4, 256),
                                nn.PReLU(),
                                nn.Linear(256, 256),
                                nn.PReLU(),
                                nn.Linear(256, 2)
                                )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output


class ResnetEmbeddingNet(nn.Module):
    def __init__(self):
        super(ResnetEmbeddingNet, self).__init__()
        self.fc = nn.Sequential(nn.Linear(2048, 256),
                                nn.PReLU(),
                                nn.Linear(256, 256),
                                nn.PReLU(),
                                nn.Linear(256, 2)
                                )

    def forward(self, x):
        output = self.fc(x)
        return output


class TripletNetwork(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNetwork, self).__init__()

        self.emb_net = embedding_net

    def forward(self, target, positive, negative):
        emb_target = self.emb_net(target)
        emb_positive = self.emb_net(positive)
        emb_negative = self.emb_net(negative)

        return emb_target, emb_positive, emb_negative

    def get_embedding(self, x):
        return self.emb_net(x)


class TripletLoss(nn.Module):
    def __init__(self, margin=1):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, target, positive, negative):
        ap_distances = F.pairwise_distance(target, positive)
        an_distances = F.pairwise_distance(target, negative)
        pn_distances = F.pairwise_distance(positive, negative)

        min_neg_dist = torch.min(an_distances, pn_distances)
        losses = F.relu(ap_distances - min_neg_dist + self.margin)

        return losses.mean()


class OnlineTripletLoss(nn.Module):
    def __init__(self, margin=1):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin

    def get_triplet_mask(self, target_idx, positive_idxs, negative_idxs):
        positive_idxs_l = []
        for pos_idx in positive_idxs:
            positive_idxs_l.append([i for i, val in enumerate(pos_idx) if val])

        positive_idxs = np.array(positive_idxs_l)

        negative_idxs_l = []
        for neg_idx in negative_idxs:
            negative_idxs_l.append([i for i, val in enumerate(neg_idx) if val])
        negative_idxs = np.array(negative_idxs_l)

        # Extract possible triplet based on computed embedding and possible positive/negative
        # return mask[i, j, k] where i=anchor, j=positive, k=negative == True if candidate
        batch_size = len(target_idx)
        pos_candidate = []
        neg_candidate = []
        for idx, targ in enumerate(target_idx):
            pos_candidate_tmp = []
            for targ_pos in positive_idxs[idx]:
                ind_pos = (targ_pos == target_idx).nonzero()
                if ind_pos.nelement() > 0:
                    pos_candidate_tmp.append(ind_pos)
            pos_candidate.append(pos_candidate_tmp)

            neg_candidate_tmp = []
            for targ_neg in negative_idxs[idx]:
                ind_neg = (targ_neg == target_idx).nonzero()
                if ind_neg.nelement() > 0:
                    neg_candidate_tmp.append(ind_neg)
            neg_candidate.append(neg_candidate_tmp)

        mask = np.zeros((batch_size, batch_size, batch_size), dtype=bool)
        # for idx, targ in enumerate(target_idx):
        #     for pos_cand in pos_candidate[idx]:
        #         for neg_cand in neg_candidate[idx]:
        #             if targ != pos_cand and targ != neg_cand and pos_cand != neg_cand:
        #                 mask[idx, pos_cand, neg_cand] = True
        return mask, pos_candidate, neg_candidate

    def forward(self, embedding, target_idx, positive_idxs, negative_idxs):
        batch_size = len(target_idx)

        with torch.no_grad():
            # Get possible triplet from computed embeddings
            mask, pos_candidate, neg_candidate = self.get_triplet_mask(
                target_idx, positive_idxs, negative_idxs)

            # pos_condidate contains indices of embedding for positive candidate

            # Compute distance between every embedding
            distance_matrix = torch.zeros(
                embedding.size()[0], embedding.size()[0])

            for idx_src, emb_src in enumerate(embedding):
                for idx_dst, emb_dst in enumerate(embedding):
                    distance_matrix[idx_src, idx_dst] = F.pairwise_distance(
                        emb_src.unsqueeze(0), emb_dst.unsqueeze(0))

            hardest_positive = None
            hardest_negative = None
            # Get hardest positive/negative
            for t_idx, _ in enumerate(target_idx):
                # Positive candidate
                pos_ind = torch.LongTensor([-1])
                ap_distance = None
                if pos_candidate[t_idx]:
                    dist = distance_matrix[t_idx, pos_candidate[t_idx]]
                    val, pos_ind = torch.max(dist, dim=0)
                    ap_distance = val
                    pos_ind = pos_ind.unsqueeze(0)
                # Negative candidate
                neg_ind = torch.LongTensor([-1])
                if pos_candidate[t_idx] and neg_candidate[t_idx]:
                    # dist = ap_distance - \
                    #     distance_matrix[t_idx,
                    #                     neg_candidate[t_idx]] + self.margin
                    dist = distance_matrix[t_idx, neg_candidate[t_idx]]
                    val, neg_ind = torch.min(dist, dim=0)
                    neg_ind = neg_ind.unsqueeze(0)
                if hardest_positive is None:
                    hardest_positive = pos_ind
                else:
                    hardest_positive = torch.cat((hardest_positive, pos_ind))

                if hardest_negative is None:
                    hardest_negative = neg_ind
                else:
                    hardest_negative = torch.cat((hardest_negative, neg_ind))

        # shape(hardest_positive) = len(target_idx, 1) Hardest per anchor
        # hardest_positive contains indices of hardest pos_candidate or -1 if no pos_candidate
        valid_idx = [t_idx for t_idx, _ in enumerate(
            target_idx) if hardest_positive[t_idx] != -1 and hardest_negative[t_idx] != -1]

        # for v in valid_idx:
        #     pos_cand = hardest_positive[v]
        #     pos_indx = target_idx[pos_candidate[v][pos_cand]]

        #     print("Positive candidate", pos_cand)
        #     print("Positive index", pos_indx)
        #     print(target_idx)
        #     plt.figure("Anchor")
        #     plt.imshow(self.h5_file["observation"][target_idx[v]])
        #     plt.figure("Positive")
        #     print(np.shape(pos_candidate))
        #     print(self.h5_file["shortest_path_distance"]
        #           [target_idx[v]][pos_indx])
        #     plt.imshow(self.h5_file["observation"][pos_indx])
        #     plt.show()
        #     exit()

        anc_emb = embedding[valid_idx]

        hardest_pos_ind_embedding = []
        hardest_neg_ind_embedding = []
        for v in valid_idx:
            hardest_pos_ind_embedding.append(
                pos_candidate[v][hardest_positive[v]].squeeze())
            hardest_neg_ind_embedding.append(
                neg_candidate[v][hardest_negative[v]].squeeze())

        # From positive candidate pos_candidate[valid_idx]
        # get the hardest [hardest_positive[valid_idx]]
        pos_emb = embedding[hardest_pos_ind_embedding]

        # From negative candidate neg_candidate[valid_idx]
        # get the hardest [hardest_negative[valid_idx]]
        neg_emb = embedding[hardest_neg_ind_embedding]

        pdist = nn.PairwiseDistance(p=2)
        ap_distances = pdist(anc_emb, pos_emb)
        an_distances = pdist(anc_emb, neg_emb)
        pn_distances = pdist(pos_emb, neg_emb)

        min_neg_dist = torch.min(an_distances, pn_distances)
        loss = F.relu(
            ap_distances - min_neg_dist + self.margin)

        return loss.mean()
