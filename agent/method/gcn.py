import torch

from torchvision import transforms

from torchvision import transforms
from .abs_method import AbstractMethod


class GCN(AbstractMethod):
    def forward_policy(self, env, device, policy_networks):
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                0.229, 0.224, 0.225])])
        state = {
            "current": env.render('resnet_features'),
            "goal": env.render_target('word_features'),
            "observation": normalize(env.observation).unsqueeze(0),
        }

        x_processed = torch.from_numpy(state["current"])
        goal_processed = torch.from_numpy(state["goal"])
        obs = state['observation']

        x_processed = x_processed.to(self.device)
        goal_processed = goal_processed.to(self.device)
        obs = obs.to(self.device)

        (policy, value) = self.policy_networks(
            (x_processed, goal_processed, obs,))

        return policy, value, state
