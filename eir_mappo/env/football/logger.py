from eir_mappo.common.base_logger import BaseLogger
import numpy as np


class FootballLogger(BaseLogger):
    def __init__(self, args, algo_args, env_args, num_agents, writter, run_dir):
        super().__init__(args, algo_args, env_args, num_agents, writter, run_dir)
        self.best_eval_avg_rew = -1e10

    def episode_log(self, actor_train_infos, critic_train_info, actor_buffer, critic_buffer):
        super().episode_log(actor_train_infos, critic_train_info, actor_buffer, critic_buffer)
        print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
              .format(self.env_args["env_name"],
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

    def eval_init(self):
        self.total_num_steps = self.episode * \
            self.algo_args["train"]["episode_length"] * \
            self.algo_args["train"]["n_rollout_threads"]
        self.eval_episode_rewards = []
        self.one_episode_rewards = []
        for eval_i in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
            self.one_episode_rewards.append([])
            self.eval_episode_rewards.append([])
        self.eval_episode_cnt = 0
        self.eval_score_cnt = 0

    def eval_per_step(self, eval_data):
        eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = eval_data
        for eval_i in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
            self.one_episode_rewards[eval_i].append(eval_rewards[eval_i])
        self.eval_infos = eval_infos

    def eval_thread_done(self, tid):
        self.eval_episode_rewards[tid].append(
            np.sum(self.one_episode_rewards[tid], axis=0))
        self.one_episode_rewards[tid] = []
        self.eval_episode_cnt += 1
        if self.eval_infos[tid][0]["score_reward"] > 0:
            self.eval_score_cnt += 1

    def eval_log(self, eval_episode):
        super().eval_log(eval_episode)
        eval_score_rate = self.eval_score_cnt / self.eval_episode_cnt
        eval_env_infos = {'eval_return_mean': self.eval_episode_rewards,
                          'eval_return_std': [np.std(self.eval_episode_rewards)],
                          'eval_score_rate': [eval_score_rate]}
        self.log_env(eval_env_infos)
        eval_avg_rew = np.mean(self.eval_episode_rewards)
        print("eval_average_episode_rewards is {}, eval_score_rate is {}.".format(
            eval_avg_rew, eval_score_rate))
        self.log_file.write(",".join(map(str, [self.total_num_steps, eval_avg_rew, eval_score_rate])) + "\n")
        self.log_file.flush()
        return True
        # if self.best_eval_avg_rew < eval_avg_rew:
        #     self.best_eval_avg_rew = eval_avg_rew
        #     return True
        # else:
        #     return False

    def eval_log_adv(self, eval_episode, agent_id):
        super().eval_log(eval_episode)
        eval_score_rate = self.eval_score_cnt / self.eval_episode_cnt
        eval_env_infos = {'eval_adv{}_return_mean'.format(agent_id): self.eval_episode_rewards,
                          'eval_adv{}_return_std'.format(agent_id): [np.std(self.eval_episode_rewards)],
                          'eval_adv{}_score_rate'.format(agent_id): [eval_score_rate]}
        self.log_env(eval_env_infos)
        eval_avg_rew = np.mean(self.eval_episode_rewards)
        print("eval_adv{}_average_episode_rewards is {}, eval_adv{}_score_rate is {}.".format(
            agent_id, eval_avg_rew, agent_id, eval_score_rate))