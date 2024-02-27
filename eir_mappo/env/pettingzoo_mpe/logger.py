from eir_mappo.common.base_logger import BaseLogger
import numpy as np


class PettingZooMPELogger(BaseLogger):
    def __init__(self, args, algo_args, env_args, num_agents, writter, run_dir):
        super().__init__(args, algo_args, env_args, num_agents, writter, run_dir)
        self.best_eval_avg_rew = -1e10

    def episode_log(self, actor_train_infos, critic_train_info, actor_buffer, critic_buffer):
        super().episode_log(actor_train_infos, critic_train_info, actor_buffer, critic_buffer)
        print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
              .format(self.env_args["scenario"],
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

    def eval_log(self, eval_episode):
        super().eval_log(eval_episode)
        eval_env_infos = {'eval_return_mean': self.eval_episode_rewards,
                          'eval_return_std': [np.std(self.eval_episode_rewards)]}
        self.log_env(eval_env_infos)
        eval_avg_rew = np.mean(self.eval_episode_rewards)
        print("eval_average_episode_rewards is {}.".format(
            eval_avg_rew))
        self.log_file.write(",".join(map(str, [self.total_num_steps, eval_avg_rew])) + "\n")
        self.log_file.flush()
        return True
        # if self.best_eval_avg_rew < eval_avg_rew:
        #     self.best_eval_avg_rew = eval_avg_rew
        #     return True
        # else:
        #     return False

    def eval_log_adv(self, eval_episode, agent_id):
        super().eval_log(eval_episode)
        eval_env_infos = {'eval_adv{}_return_mean'.format(agent_id): self.eval_episode_rewards,
                          'eval_adv{}_return_std'.format(agent_id): [np.std(self.eval_episode_rewards)]}
        self.log_env(eval_env_infos)
        eval_avg_rew = np.mean(self.eval_episode_rewards)
        print("eval_adv{}_average_episode_rewards is {}.".format(agent_id, eval_avg_rew))
