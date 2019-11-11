import torch

from .abs_method import AbstractMethod


class SimilarityGrid(AbstractMethod):

    def extract_input(self, env, device):
        state = {
            "current": env.render('resnet_features'),
            "goal": env.render_target('word_features')
        }

        if self.method == 'word2vec' or self.method == 'word2vec_noconv':
            state["object_mask"] = env.render_mask_similarity()
            x_processed = torch.from_numpy(state["current"])
            goal_processed = torch.from_numpy(state["goal"])
            object_mask = torch.from_numpy(state['object_mask'])

            x_processed = x_processed.to(device)
            goal_processed = goal_processed.to(device)
            object_mask = object_mask.to(device)

            return state, x_processed, goal_processed, object_mask
        elif self.method == 'word2vec_notarget':
            state["object_mask"] = env.render_mask_similarity()
            x_processed = torch.from_numpy(state["current"])
            object_mask = torch.from_numpy(state['object_mask'])

            x_processed = x_processed.to(device)
            object_mask = object_mask.to(device)

            return state, x_processed, object_mask

        elif self.method == 'word2vec_nosimi':
            x_processed = torch.from_numpy(state["current"])
            goal_processed = torch.from_numpy(state["goal"])

            x_processed = x_processed.to(device)
            goal_processed = goal_processed.to(device)

            return state, x_processed, goal_processed

        elif self.method == 'word2vec_notarget_lstm':
            state["object_mask"] = env.render_mask_similarity()
            state["hidden"] = env.render_hidden_state()
            
            # Change current state with only last frame
            state["current"] = state["current"][:, -1]
            x_processed = torch.from_numpy(state["current"])
            object_mask = torch.from_numpy(state['object_mask'])
            h1, c1 = state['hidden']

            x_processed = x_processed.to(device)
            object_mask = object_mask.to(device)
            h1 = h1.to(device)
            c1 = c1.to(device)
            hidden = (h1, c1)

            return state, x_processed, object_mask, hidden


    def forward_policy(self, env, device, policy_networks):
        if self.method == 'word2vec' or self.method == 'word2vec_noconv':
            state, x_processed, goal_processed, object_mask = self.extract_input(env, device)
            (policy, value) = policy_networks((x_processed, goal_processed, object_mask,))

        elif self.method == 'word2vec_notarget':
            state, x_processed, object_mask = self.extract_input(env, device)
            (policy, value) = policy_networks((x_processed, object_mask,))

        elif self.method == 'word2vec_nosimi':
            state, x_processed, goal_processed = self.extract_input(env, device)
            (policy, value) = policy_networks((x_processed, goal_processed,))

        elif self.method == 'word2vec_notarget_lstm':
            state, x_processed, object_mask, hidden = self.extract_input(env, device)

            # Save current hidden value
            outputs= []
            def hook(module, input, output):
                outputs.append(output)

            handle = policy_networks[0].net.lstm.register_forward_hook(hook)
            (policy, value) = policy_networks((x_processed, object_mask, hidden))
            handle.remove()

            env.set_hidden(tuple([o.detach() for o in outputs[-1]]))

        return policy, value, state
