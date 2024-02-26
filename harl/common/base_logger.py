import time
import numpy as np


class BaseLogger:
    def __init__(self, args, algo_args, env_args, num_agents, writter, run_dir):
        self.args = args
        self.algo_args = algo_args
        self.env_args = env_args
        self.num_agents = num_agents
        self.writter = writter
        self.run_dir = run_dir
        self.log_file = open(str(run_dir / "progress.txt"), "w")

    def init(self, episodes):
        self.start = time.time()
        self.episodes = episodes

    def episode_init(self, episode):
        self.episode = episode

    def per_step(self, data):
        pass

    def episode_log(self, actor_train_infos, critic_train_info, actor_buffer, critic_buffer):
        self.total_num_steps = self.episode * \
            self.algo_args["train"]["episode_length"] * \
            self.algo_args["train"]["n_rollout_threads"]
        self.end = time.time()

    def eval_init(self):
        self.total_num_steps = self.episode * \
            self.algo_args["train"]["episode_length"] * \
            self.algo_args["train"]["n_rollout_threads"]
        self.eval_episode_rewards = []
        self.one_episode_rewards = []
        for eval_i in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
            self.one_episode_rewards.append([])
            self.eval_episode_rewards.append([])

    def eval_per_step(self, eval_data):
        eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = eval_data
        for eval_i in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
            self.one_episode_rewards[eval_i].append(eval_rewards[eval_i])

    def eval_thread_done(self, tid):
        self.eval_episode_rewards[tid].append(
            np.sum(self.one_episode_rewards[tid], axis=0)[0, 0])
        self.one_episode_rewards[tid] = []

    def eval_log(self, eval_episode):
        self.eval_episode_rewards = np.concatenate(
            [rewards for rewards in self.eval_episode_rewards if rewards])

    def log_train(self, actor_train_infos, critic_train_info):
        # log actor
        for agent_id in range(self.num_agents):
            for k, v in actor_train_infos[agent_id].items():
                agent_k = "agent%i/" % agent_id + k
                self.writter.add_scalar(
                    agent_k, v, self.total_num_steps)
        # log critic
        for k, v in critic_train_info.items():
            critic_k = "critic/" + k
            self.writter.add_scalar(
                critic_k, v, self.total_num_steps)

    def log_env(self, env_infos):
        for k, v in env_infos.items():
            if len(v) > 0:
                self.writter.add_scalar(
                    "env/" + k, np.mean(v), self.total_num_steps)

    def close(self):
        self.log_file.close()