import torch

from .abs_method import AbstractMethod


class TargetDriven(AbstractMethod):
    def forward_policy(self, env, device, policy_networks):
        state = {
            "current": env.render('resnet_features'),
            "goal": env.render_target('resnet_features'),
        }

        x_processed = torch.from_numpy(state["current"])
        goal_processed = torch.from_numpy(state["goal"])

        x_processed = x_processed.to(device)
        goal_processed = goal_processed.to(device)

        (policy, value) = policy_networks(
            (x_processed, goal_processed,))
        return policy, value, state
