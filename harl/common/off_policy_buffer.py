from harl.util.util import get_shape_from_obs_space, get_shape_from_act_space
import numpy as np
import torch


class OffPolicyBuffer:
    def __init__(self, args, share_obs_space, num_agents, obs_spaces, act_spaces):
        self.buffer_size = args["buffer_size"]
        self.batch_size = args["batch_size"]
        self.n_step = args["n_step"]
        self.n_rollout_threads = args["n_rollout_threads"]
        self.gamma = args["gamma"]
        self.cur_size = 0
        self.idx = 0
        self.num_agents = num_agents
        share_obs_shape = get_shape_from_obs_space(share_obs_space)
        if type(share_obs_shape[-1]) == list:
            share_obs_shape = share_obs_shape[:1]
        obs_shapes = []
        act_shapes = []
        for agent_id in range(num_agents):
            obs_shape = get_shape_from_obs_space(obs_spaces[agent_id])
            if type(obs_shape[-1]) == list:
                obs_shape = obs_shape[:1]
            obs_shapes.append(obs_shape)
            act_shape = get_shape_from_act_space(act_spaces[agent_id])
            if act_spaces[agent_id].__class__.__name__ == "Discrete":
                act_shape = act_spaces[agent_id].n
            act_shapes.append(act_shape)

        self.share_obs = np.zeros(
            (self.buffer_size, *share_obs_shape), dtype=np.float32)
        self.next_share_obs = np.zeros(
            (self.buffer_size, *share_obs_shape), dtype=np.float32)

        self.obs = []
        self.next_obs = []
        self.available_actions = []
        self.next_available_actions = []
        for agent_id in range(num_agents):
            self.obs.append(np.zeros(
                (self.buffer_size, *obs_shapes[agent_id]), dtype=np.float32))
            self.next_obs.append(np.zeros(
                (self.buffer_size, *obs_shapes[agent_id]), dtype=np.float32))
            self.available_actions.append(np.ones(
                (self.buffer_size, act_shapes[agent_id]), dtype=np.float32))
            self.next_available_actions.append(np.ones(
                (self.buffer_size, act_shapes[agent_id]), dtype=np.float32))

        self.rewards = np.zeros(
            (self.buffer_size, 1), dtype=np.float32)

        self.actions = []
        
        for agent_id in range(num_agents):
            self.actions.append(np.zeros(
                (self.buffer_size, act_shapes[agent_id]), dtype=np.float32))

        self.dones = np.full((self.buffer_size, 1), False)
        self.terms = np.full((self.buffer_size, 1), False)

    def insert(self, data):
        """
        share_obs: (n_rollout_threads, *share_obs_shape)
        obs: [(n_rollout_threads, *obs_shapes[agent_id]) for agent_id in range(num_agents)]
        action: [(n_rollout_threads, *act_shapes[agent_id]) for agent_id in range(num_agents)]
        reward: (n_rollout_threads, 1)
        done: (n_rollout_threads, 1)
        term: (n_rollout_threads, 1)
        next_share_obs: (n_rollout_threads, *share_obs_shape)
        next_obs: [(n_rollout_threads, *obs_shapes[agent_id]) for agent_id in range(num_agents)]
        """
        share_obs, obs, available_actions, actions, reward, done, term, next_share_obs, next_obs, next_available_actions = data
        length = share_obs.shape[0]
        if self.idx + length <= self.buffer_size:
            s = self.idx
            e = self.idx + length
            self.share_obs[s:e] = share_obs.copy()
            self.rewards[s:e] = reward.copy()
            self.dones[s:e] = done.copy()
            self.terms[s:e] = term.copy()
            self.next_share_obs[s:e] = next_share_obs.copy()
            for agent_id in range(self.num_agents):
                self.obs[agent_id][s:e] = obs[agent_id].copy()
                self.actions[agent_id][s:e] = actions[agent_id].copy()
                self.next_obs[agent_id][s:e] = next_obs[agent_id].copy()
                if available_actions is not None:
                    self.available_actions[agent_id][s:e] = available_actions[agent_id].copy()
                if next_available_actions is not None:
                    self.next_available_actions[agent_id][s:e] = next_available_actions[agent_id].copy()
        else:
            len1 = self.buffer_size - self.idx
            len2 = length - len1
            s = self.idx
            e = self.buffer_size
            self.share_obs[s:e] = share_obs[0:len1].copy()
            self.rewards[s:e] = reward[0:len1].copy()
            self.dones[s:e] = done[0:len1].copy()
            self.terms[s:e] = term[0:len1].copy()
            self.next_share_obs[s:e] = next_share_obs[0:len1].copy()
            for agent_id in range(self.num_agents):
                self.obs[agent_id][s:e] = obs[agent_id][0:len1].copy()
                self.actions[agent_id][s:e] = actions[agent_id][0:len1].copy()
                self.next_obs[agent_id][s:e] = next_obs[agent_id][0:len1].copy()
                if available_actions is not None:
                    self.available_actions[agent_id][s:e] = available_actions[0:len1].copy()
                if next_available_actions is not None:
                    self.next_available_actions[agent_id][s:e] = next_available_actions[0:len1].copy()
            s = 0
            e = len2
            self.share_obs[s:e] = share_obs[len1:length].copy()
            self.rewards[s:e] = reward[len1:length].copy()
            self.dones[s:e] = done[len1:length].copy()
            self.terms[s:e] = term[len1:length].copy()
            self.next_share_obs[s:e] = next_share_obs[len1:length].copy()
            for agent_id in range(self.num_agents):
                self.obs[agent_id][s:e] = obs[agent_id][len1:length].copy()
                self.actions[agent_id][s:e] = actions[agent_id][len1:length].copy()
                self.next_obs[agent_id][s:e] = next_obs[agent_id][len1:length].copy()
                if available_actions is not None:
                    self.available_actions[agent_id][s:e] = available_actions[len1:length].copy()
                if next_available_actions is not None:
                    self.next_available_actions[agent_id][s:e] = next_available_actions[len1:length].copy()
        self.idx = (self.idx + length) % self.buffer_size
        self.cur_size = min(self.cur_size + length, self.buffer_size)

    def sample(self):
        """"
        sample data for training.
        returns:
        sp_share_obs: (batch_size, dim)
        sp_obs: (n_agents, batch_size, dim)
        sp_actions: (n_agents, batch_size, dim)
        sp_reward: (batch_size, 1)
        sp_done: (batch_size, 1)
        sp_term: (batch_size, 1)
        sp_next_share_obs: (batch_size, dim)
        sp_next_obs: (n_agents, batch_size, dim)
        """
        self.update_end_flag()
        indice = torch.randperm(self.cur_size).numpy()[:self.batch_size]
        sp_share_obs = self.share_obs[indice]
        sp_obs = np.array([self.obs[agent_id][indice]
                           for agent_id in range(self.num_agents)])
        sp_actions = np.array([self.actions[agent_id][indice]
                               for agent_id in range(self.num_agents)])
        sp_available_actions = np.array([self.available_actions[agent_id][indice]
                                        for agent_id in range(self.num_agents)])
        indices = [indice]
        for _ in range(self.n_step - 1):
            indices.append(self.next(indices[-1]))
        sp_done = self.dones[indices[-1]]
        sp_term = self.terms[indices[-1]]
        sp_next_share_obs = self.next_share_obs[indices[-1]]
        sp_next_obs = np.array([self.next_obs[agent_id][indices[-1]] for agent_id in range(self.num_agents)])
        sp_next_available_actions = np.array([self.next_available_actions[agent_id][indices[-1]] for agent_id in range(self.num_agents)])
        gamma_buffer = np.ones(self.n_step + 1)
        for i in range(1, self.n_step + 1):
            gamma_buffer[i] = gamma_buffer[i - 1] * self.gamma
        sp_reward = np.zeros((self.batch_size, 1))
        gammas = np.full(self.batch_size, self.n_step)
        for n in range(self.n_step - 1, -1, -1):
            now = indices[n]
            gammas[self.end_flag[now] > 0] = n + 1
            sp_reward[self.end_flag[now] > 0] = 0.0
            sp_reward = self.rewards[now] + self.gamma * sp_reward
        sp_gamma = gamma_buffer[gammas].reshape(self.batch_size, 1)
        return sp_share_obs, sp_obs, sp_available_actions, sp_actions, sp_reward, sp_done, sp_term, sp_next_share_obs, sp_next_obs, sp_next_available_actions, sp_gamma

    def next(self, indices):
        return (indices + (1 - self.end_flag[indices]) * self.n_rollout_threads) % self.buffer_size

    def update_end_flag(self):
        # self.unfinished_index = []
        # for i in range(self.n_rollout_threads):
        #     self.unfinished_index.append((self.idx - i - 1 + self.cur_size) % self.cur_size)
        self.unfinished_index = (self.idx - np.arange(self.n_rollout_threads) - 1 + self.cur_size) % self.cur_size
        self.end_flag = self.dones.copy().squeeze()
        self.end_flag[self.unfinished_index] = True

    def get_mean_rewards(self):
        return np.mean(self.rewards[:self.cur_size])