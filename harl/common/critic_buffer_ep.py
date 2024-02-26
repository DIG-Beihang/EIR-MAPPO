import torch
import numpy as np
from collections import defaultdict
from harl.util.util import check, get_shape_from_obs_space, _flatten, _sa_cast


class CriticBufferEP:
    def __init__(self, args, share_obs_space):
        self.episode_length = args["episode_length"]
        self.n_rollout_threads = args["n_rollout_threads"]
        self.hidden_sizes = args["hidden_sizes"]
        self.rnn_hidden_size = self.hidden_sizes[-1]
        self.recurrent_N = args["recurrent_N"]
        self.gamma = args["gamma"]
        self.gae_lambda = args["gae_lambda"]
        self.use_gae = args["use_gae"]
        self.use_proper_time_limits = args["use_proper_time_limits"]

        share_obs_shape = get_shape_from_obs_space(share_obs_space)

        if type(share_obs_shape[-1]) == list:
            share_obs_shape = share_obs_shape[:1]

        self.share_obs = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, *share_obs_shape), dtype=np.float32)

        self.rnn_states_critic = np.zeros((self.episode_length + 1, self.n_rollout_threads,
                                           self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)

        self.value_preds = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, 1), dtype=np.float32)
        self.returns = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, 1), dtype=np.float32)

        self.rewards = np.zeros(
            (self.episode_length, self.n_rollout_threads, 1), dtype=np.float32)

        self.masks = np.ones(
            (self.episode_length + 1, self.n_rollout_threads, 1), dtype=np.float32)
        
        self.bad_masks = np.ones_like(self.masks)

        self.step = 0

    def insert(self, share_obs, rnn_states_critic, value_preds, rewards, masks, bad_masks):
        self.share_obs[self.step + 1] = share_obs.copy()
        self.rnn_states_critic[self.step + 1] = rnn_states_critic.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()
        self.bad_masks[self.step + 1] = bad_masks.copy()

        self.step = (self.step + 1) % self.episode_length

    def after_update(self):
        self.share_obs[0] = self.share_obs[-1].copy()
        self.rnn_states_critic[0] = self.rnn_states_critic[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()

    def get_mean_rewards(self):
        return np.mean(self.rewards)

    def compute_returns(self, next_value, value_normalizer=None):
        """
        Compute returns either as discounted sum of rewards, or using GAE.
        :param next_value: (np.ndarray) value predictions for the step after the last episode step.
        :param value_normalizer: (PopArt) If not None, PopArt value normalizer instance.
        """
        if self.use_proper_time_limits:
            if self.use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    if value_normalizer is not None:
                        delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(
                            self.value_preds[step + 1]) * self.masks[step + 1] - value_normalizer.denormalize(self.value_preds[step])
                        gae = delta + self.gamma * \
                            self.gae_lambda * self.masks[step + 1] * gae
                        gae = self.bad_masks[step + 1] * gae
                        self.returns[step] = gae + \
                            value_normalizer.denormalize(
                                self.value_preds[step])
                    else:
                        delta = self.rewards[step] + self.gamma * self.value_preds[step +
                                                                                1] * self.masks[step + 1] - self.value_preds[step]
                        gae = delta + self.gamma * \
                            self.gae_lambda * self.masks[step + 1] * gae
                        gae = self.bad_masks[step + 1] * gae
                        self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    if value_normalizer is not None:
                        self.returns[step] = (self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[
                            step]) * self.bad_masks[step + 1] \
                                             + (1 - self.bad_masks[step + 1]) * value_normalizer.denormalize(
                            self.value_preds[step])
                    else:
                        self.returns[step] = (self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[
                            step]) * self.bad_masks[step + 1] \
                                             + (1 - self.bad_masks[step + 1]) * self.value_preds[step]
        else:
            if self.use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    if value_normalizer is not None:
                        delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(
                            self.value_preds[step + 1]) * self.masks[step + 1] - value_normalizer.denormalize(self.value_preds[step])
                        gae = delta + self.gamma * \
                            self.gae_lambda * self.masks[step + 1] * gae
                        self.returns[step] = gae + \
                            value_normalizer.denormalize(
                                self.value_preds[step])
                    else:
                        delta = self.rewards[step] + self.gamma * self.value_preds[step +
                                                                                1] * self.masks[step + 1] - self.value_preds[step]
                        gae = delta + self.gamma * \
                            self.gae_lambda * self.masks[step + 1] * gae
                        self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    self.returns[step] = self.returns[step + 1] * \
                        self.gamma * self.masks[step + 1] + self.rewards[step]

    def feed_forward_generator_critic(self, critic_num_mini_batch=None, mini_batch_size=None):
        episode_length, n_rollout_threads = self.rewards.shape[0:2]
        batch_size = n_rollout_threads * episode_length

        if mini_batch_size is None:
            assert batch_size >= critic_num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(n_rollout_threads, episode_length, n_rollout_threads * episode_length,
                          critic_num_mini_batch))
            mini_batch_size = batch_size // critic_num_mini_batch

        rand = torch.randperm(batch_size).numpy()
        sampler = [rand[i*mini_batch_size:(i+1)*mini_batch_size]
                   for i in range(critic_num_mini_batch)]

        share_obs = self.share_obs[:-1].reshape(-1, *self.share_obs.shape[2:])
        rnn_states_critic = self.rnn_states_critic[:-
                                                   1].reshape(-1, *self.rnn_states_critic.shape[2:])
        value_preds = self.value_preds[:-1].reshape(-1, 1)
        returns = self.returns[:-1].reshape(-1, 1)
        masks = self.masks[:-1].reshape(-1, 1)

        for indices in sampler:
            # obs size [T+1 N Dim]-->[T N Dim]-->[T*N,Dim]-->[index,Dim]
            share_obs_batch = share_obs[indices]
            rnn_states_critic_batch = rnn_states_critic[indices]
            value_preds_batch = value_preds[indices]
            return_batch = returns[indices]
            masks_batch = masks[indices]

            yield share_obs_batch, rnn_states_critic_batch, value_preds_batch, return_batch, masks_batch

    def naive_recurrent_generator_critic(self, critic_num_mini_batch):
        n_rollout_threads = self.rewards.shape[1]
        assert n_rollout_threads >= critic_num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(n_rollout_threads, critic_num_mini_batch))
        num_envs_per_batch = n_rollout_threads // critic_num_mini_batch
        perm = torch.randperm(n_rollout_threads).numpy()
        for start_ind in range(0, n_rollout_threads, num_envs_per_batch):
            share_obs_batch = []
            rnn_states_critic_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                share_obs_batch.append(self.share_obs[:-1, ind])
                rnn_states_critic_batch.append(
                    self.rnn_states_critic[0:1, ind])
                value_preds_batch.append(self.value_preds[:-1, ind])
                return_batch.append(self.returns[:-1, ind])
                masks_batch.append(self.masks[:-1, ind])

            # [N[T, dim]]
            T, N = self.episode_length, num_envs_per_batch
            # These are all from_numpys of size (T, N, -1)
            share_obs_batch = np.stack(share_obs_batch, 1)
            value_preds_batch = np.stack(value_preds_batch, 1)
            return_batch = np.stack(return_batch, 1)
            masks_batch = np.stack(masks_batch, 1)

            # States is just a (N, -1) from_numpy [N[1,dim]]
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch, 1).reshape(
                N, *self.rnn_states_critic.shape[2:])

            # Flatten the (T, N, ...) from_numpys to (T * N, ...)
            share_obs_batch = _flatten(T, N, share_obs_batch)
            value_preds_batch = _flatten(T, N, value_preds_batch)
            return_batch = _flatten(T, N, return_batch)
            masks_batch = _flatten(T, N, masks_batch)

            yield share_obs_batch, rnn_states_critic_batch, value_preds_batch, return_batch, masks_batch

    def recurrent_generator_critic(self, critic_num_mini_batch, data_chunk_length):
        episode_length, n_rollout_threads = self.rewards.shape[0:2]
        batch_size = n_rollout_threads * episode_length
        data_chunks = batch_size // data_chunk_length  # [C=r*T/L]
        mini_batch_size = data_chunks // critic_num_mini_batch

        assert episode_length * n_rollout_threads >= data_chunk_length, (
            "PPO requires the number of processes ({}) * episode length ({}) "
            "to be greater than or equal to the number of "
            "data chunk length ({}).".format(n_rollout_threads, episode_length, data_chunk_length))
        assert data_chunks >= 2, ("need larger batch size")

        rand = torch.randperm(data_chunks).numpy()
        sampler = [rand[i*mini_batch_size:(i+1)*mini_batch_size]
                   for i in range(critic_num_mini_batch)]

        if len(self.share_obs.shape) > 3:
            share_obs = self.share_obs[:-1].transpose(
                1, 0, 2, 3, 4).reshape(-1, *self.share_obs.shape[2:])
        else:
            share_obs = _sa_cast(self.share_obs[:-1])

        value_preds = _sa_cast(self.value_preds[:-1])
        returns = _sa_cast(self.returns[:-1])
        masks = _sa_cast(self.masks[:-1])
        rnn_states_critic = self.rnn_states_critic[:-1].transpose(
            1, 0, 2, 3).reshape(-1, *self.rnn_states_critic.shape[2:])

        for indices in sampler:
            share_obs_batch = []
            rnn_states_critic_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            for index in indices:
                ind = index * data_chunk_length
                # size [T+1 N M Dim]-->[T N Dim]-->[N T Dim]-->[T*N,Dim]-->[L,Dim]
                share_obs_batch.append(share_obs[ind:ind+data_chunk_length])
                value_preds_batch.append(
                    value_preds[ind:ind+data_chunk_length])
                return_batch.append(returns[ind:ind+data_chunk_length])
                masks_batch.append(masks[ind:ind+data_chunk_length])
                # size [T+1 N Dim]-->[T N Dim]-->[T*N,Dim]-->[1,Dim]
                rnn_states_critic_batch.append(rnn_states_critic[ind])
            L, N = data_chunk_length, mini_batch_size

            # These are all from_numpys of size (N, L, Dim)
            share_obs_batch = np.stack(share_obs_batch, axis=1)

            value_preds_batch = np.stack(value_preds_batch, axis=1)
            return_batch = np.stack(return_batch, axis=1)
            masks_batch = np.stack(masks_batch, axis=1)

            # States is just a (N, -1) from_numpy
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch).reshape(
                N, *self.rnn_states_critic.shape[2:])

            # Flatten the (L, N, ...) from_numpys to (L * N, ...)
            share_obs_batch = _flatten(L, N, share_obs_batch)
            value_preds_batch = _flatten(L, N, value_preds_batch)
            return_batch = _flatten(L, N, return_batch)
            masks_batch = _flatten(L, N, masks_batch)
            yield share_obs_batch, rnn_states_critic_batch, value_preds_batch, return_batch, masks_batch
