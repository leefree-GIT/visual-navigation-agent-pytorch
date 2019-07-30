import torch

from .abs_method import AbstractMethod


class SimilarityGrid(AbstractMethod):
    def forward_policy(self, env, device, policy_networks):
        state = {
            "current": env.render('resnet_features'),
            "goal": env.render_target('word_features'),
            "object_mask": env.render_mask_similarity()
        }

        if self.method == 'word2vec' or self.method == 'word2vec_noconv':
            x_processed = torch.from_numpy(state["current"])
            goal_processed = torch.from_numpy(state["goal"])
            object_mask = torch.from_numpy(state['object_mask'])

            x_processed = x_processed.to(device)
            goal_processed = goal_processed.to(device)
            object_mask = object_mask.to(device)
            (policy, value) = policy_networks(
                (x_processed, goal_processed, object_mask,))
        elif self.method == 'word2vec_notarget':
            x_processed = torch.from_numpy(state["current"])
            object_mask = torch.from_numpy(state['object_mask'])

            x_processed = x_processed.to(device)
            object_mask = object_mask.to(device)

            (policy, value) = policy_networks(
                (x_processed, object_mask,))
        elif self.method == 'word2vec_nosimi':
            x_processed = torch.from_numpy(state["current"])
            goal_processed = torch.from_numpy(state["goal"])

            x_processed = x_processed.to(device)
            goal_processed = goal_processed.to(device)

            (policy, value) = policy_networks(
                (x_processed, goal_processed,))
        return policy, value, state
