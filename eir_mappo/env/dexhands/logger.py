from eir_mappo.common.base_logger import BaseLogger
import numpy as np


class DexHandsLogger(BaseLogger):
    def __init__(self, args, algo_args, env_args, num_agents, writter, run_dir):
        super().__init__(args, algo_args, env_args, num_agents, writter, run_dir)

    def init(self, episodes):
        super().init(episodes)
        self.train_episode_rewards = [0 for _ in range(
            self.algo_args["train"]["n_rollout_threads"])]
        self.done_episodes_rewards = []

    def per_step(self, data):
        obs, share_obs, rewards, dones, infos, available_actions, values, actions, action_log_probs,  rnn_states, rnn_states_critic = data
        dones_env = np.all(dones, axis=1)
        reward_env = np.mean(rewards, axis=1).flatten()
        self.train_episode_rewards += reward_env
        for t in range(self.algo_args["train"]["n_rollout_threads"]):
            if dones_env[t]:
                self.done_episodes_rewards.append(
                    self.train_episode_rewards[t])
                self.train_episode_rewards[t] = 0

    def episode_log(self, actor_train_infos, critic_train_info, actor_buffer, critic_buffer):
        super().episode_log(actor_train_infos, critic_train_info, actor_buffer, critic_buffer)
        print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
              .format(self.env_args["task"],
                      self.args.algo,
                      self.args.env,
                      self.episode,
                      self.episodes,
                      self.total_num_steps,
                      self.algo_args["train"]["num_env_steps"],
                      int(self.total_num_steps / (self.end - self.start))))

        critic_train_info["average_step_rewards"] = critic_buffer.get_mean_rewards()
        self.log_train(actor_train_infos, critic_train_info)

        print("average_step_rewards is {}.".format(
            critic_train_info["average_step_rewards"]))

        if len(self.done_episodes_rewards) > 0:
            aver_episode_rewards = np.mean(self.done_episodes_rewards)
            print("some episodes done, average rewards: ", aver_episode_rewards)
            self.writter.add_scalars("train_episode_rewards", {"aver_rewards": aver_episode_rewards}, self.total_num_steps)
            self.log_file.write(",".join(map(str, [self.total_num_steps, aver_episode_rewards])) + "\n")
            self.log_file.flush()
            self.done_episodes_rewards = []