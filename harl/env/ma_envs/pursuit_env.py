from .envs.point_envs.pursuit_evasion import PursuitEvasionEnv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib

class PursuitEnv:
    def __init__(self, args):
        self.args = args
        self.env = PursuitEvasionEnv(windows_size=self.args["windows_size"],
                                    use_history=self.args["use_history"],
                                    nr_pursuers=self.args["nr_agents"],
                                    nr_evaders=self.args["nr_evaders"],
                                    obs_mode=self.args["obs_mode"],
                                    comm_radius=200 * np.sqrt(2),
                                    world_size=self.args["world_size"],
                                    distance_bins=self.args["distance_bins"],
                                    bearing_bins=self.args["bearing_bins"],
                                    torus=self.args["torus"],
                                    dynamics=self.args["dynamics"],
                                    evader_policy=self.args["evader_policy"])
        self.state_type = self.args["state_type"]
        self.n_agents = self.env.nr_agents
        self.env.reset()
        self.agents = self.env.world.agents
        if self.state_type == "EP":
            self.share_observation_space = self.unwrap(self.env.share_observation_space)
        else:
            self.share_observation_space = self.unwrap(self.env.observation_space)
        self.observation_space = self.unwrap(self.env.observation_space)
        self.action_space = self.unwrap(self.env.action_space)
        # self._seed = 0

    def step(self, actions):
        obs, s_obs, rewards, done, info, avaliable_actions = self.env.step(actions) 
        if self.state_type == "EP":
            return (
                obs,
                s_obs,
                rewards,
                self.repeat(done),
                self.repeat(info),
                avaliable_actions
            )
        else:
            return (
                obs,
                obs,
                rewards,
                self.repeat(done),
                self.repeat(info),
                avaliable_actions
            )
    
    def unwrap(self, d):
        l = []
        for i in range(self.n_agents):
            l.append(d[i])
        return l 
    
    # def wrap(self, d):
    #     l = []
    #     for i in range(self.n_agents):
    #         l.append(d[i])
    #     return l
    
    def repeat(self, a):
        return [a for _ in range(self.n_agents)]
    
    def reset(self):
        obs, s_obs, available_actions = self.env.reset()
        if self.state_type == "EP":
            return obs, s_obs, available_actions
        else:
            return obs, obs, available_actions
    
    def seed(self, seed):
        # pass
        self.env.seed(seed=seed)

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()
        
    def make_ani(self, trajectories):
        trajectories = trajectories[0]
        pos_x_list = []
        pos_y_list = []
        ori__list = []
        eva_pos_x_list = []
        eva_pos_y_list = []

        for t in trajectories:
            # print("vel:", np.mean(t["velocity"]))
            temp = t['pursuer_states'][:]
            temp_ori = temp[:, 2]
            temp_pos_x = temp[:,0]
            temp_pos_y = temp[:,1]
            pos_x_list.append(temp_pos_x)
            pos_y_list.append(temp_pos_y)
            ori__list.append(temp_ori)

            temp2 = t['evader_states'][:]
            temp_pos_x2 = temp2[:, 0]
            temp_pos_y2 = temp2[:, 1]
            eva_pos_x_list.append(temp_pos_x2)
            eva_pos_y_list.append(temp_pos_y2)
        matplotlib.use('Agg')
        print(len(pos_x_list))
        # 创建一些随机的初始数据
        show_index = 0
        x = pos_x_list[show_index]
        y = pos_y_list[show_index]
        # 创建随机的朝向向量（长度为arrow_length）
        arrow_length = 4
        angles = ori__list[show_index]
        dx = arrow_length * np.cos(angles)
        dy = arrow_length * np.sin(angles)
        # 创建一个散点图和箭头图
        fig, ax = plt.subplots()
        sc = ax.scatter(x, y)
        arrows = ax.quiver(x, y, dx, dy, angles='xy', scale_units='xy', scale=1)
        x1 = eva_pos_x_list[show_index]
        y1 = eva_pos_y_list[show_index]
        sc1 = ax.scatter(x1, y1, c = 'r',marker = 'o')  

        #  # 设置x和y轴范围
        padding = 5  # 调整范围的padding大小
        x_min = min(np.min(pos_x_list), np.min(eva_pos_x_list)) - padding
        y_min = min(np.min(pos_y_list), np.min(eva_pos_y_list)) - padding
        x_max = max(np.max(pos_x_list), np.max(eva_pos_x_list)) + padding
        y_max = max(np.max(pos_y_list), np.max(eva_pos_y_list)) + padding
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        # 更新函数，每次调用都会更新图表
        def update(frame):
            show_index = frame % len(pos_x_list)  # 使用取余运算防止越界
            x = pos_x_list[show_index]
            y = pos_y_list[show_index]
            x1 = eva_pos_x_list[show_index]
            y1 = eva_pos_y_list[show_index]

            angles = ori__list[show_index]
            show_index += 1
            dx = arrow_length * np.cos(angles)
            dy = arrow_length * np.sin(angles)

            sc.set_offsets(np.c_[x, y])
            arrows.set_offsets(np.c_[x, y])
            arrows.set_UVC(dx, dy)

            sc1.set_offsets(np.c_[x1, y1])
        ani = animation.FuncAnimation(fig, update, frames=range(len(pos_x_list)), interval=100)
        ani.save('./pursuit_animation.gif', writer='pillow')  # 保存为GIF文件
        # plt.show()