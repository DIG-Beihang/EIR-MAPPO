import numpy as np
from gym import spaces

# state: [state(1), agent_id(1), last_state(1), last_action(2), agent_adversary(2)]
class ToyExample:
    def __init__(self, args):
        self.n_agents = 2
        self.obs_last_state = args["obs_last_state"]
        self.obs_last_action = args["obs_last_action"]
        self.obs_agent_adversary = args["obs_agent_adversary"]
        self.obs_shape = 2
        if self.obs_last_state:
            self.obs_shape += 1
        if self.obs_last_action:
            self.obs_shape += 2
        if self.obs_agent_adversary:
            self.obs_shape += 2

        self.share_observation_space = [spaces.Box(low=0, high=1, shape=(self.obs_shape,), dtype="float32")]
        self.observation_space = [spaces.Box(low=0, high=1, shape=(self.obs_shape,), dtype="float32") for _ in range(2)]
        self.action_space = [spaces.Discrete(2) for _ in range(2)]

        self._state = np.random.choice([0.0, 1.0])
        self.last_state = self._state
        self._t = 0
        self.episode_limit = 100
        self.last_action = np.array([0, 0])
    
    def step(self, actions):
        if self._state == 0.0:
            self.last_state = 0.0
            if actions[0] == actions[1]:
                self._state = 1.0
                reward = 1.0
            else:
                self._state = 0.0
                reward = 0
        elif self._state == 1.0:
            self.last_state = 1.0
            if actions[0] == actions[1]:
                self._state = 0.0
                reward = 0.0
            else:
                self._state = 1.0
                reward = 1.0
        self._t += 1
        self.last_action = actions.squeeze()
        rewards = np.array([[reward] for _ in range(self.n_agents)])
        env_done = (self._t >= self.episode_limit)
        return self.get_state(), self.get_state(), rewards, \
            [env_done, env_done], [{}, {}], np.array([[1, 1] for _ in range(self.n_agents)]).astype(np.int32)
    
    def get_state(self):
        obs = np.array([[self._state, 0.0], [self._state, 1.0]])
        if self.obs_last_state:
            obs = [np.concatenate([obs[i], [self.last_state]], axis=0) for i in range(self.n_agents)]
        if self.obs_last_action:
            obs = [np.concatenate([obs[i], self.last_action], axis=0) for i in range(self.n_agents)]
        if self.obs_agent_adversary:
            obs = [np.concatenate([obs[i], [0, 0]], axis=0) for i in range(self.n_agents)]
        return np.stack(obs)

    def get_all_states(self):
        all_obs = []
        for state in [0, 1]:
            for last_state in [0, 1]:
                for last_action in [[0, 0], [0, 1], [1, 0], [1, 1]]:
                    for adv in [0, 1]:
                        obs = np.array([[state, 0.0], [state, 1.0]])
                        if self.obs_last_state:
                            obs = [np.concatenate([obs[i], [last_state]], axis=0) for i in range(self.n_agents)]
                        if self.obs_last_action:
                            obs = [np.concatenate([obs[i], last_action], axis=0) for i in range(self.n_agents)]
                        if self.obs_agent_adversary:
                            obs = [np.concatenate([obs[i], [adv, 0]], axis=0) for i in range(self.n_agents)]
                        all_obs.append(np.stack(obs))
        
        return np.stack(all_obs)

    def reset(self):
        self._state = np.random.choice([0.0, 1.0])
        self.last_state = self._state
        self._t = 0
        self.last_action = np.array([0, 0])
        return self.get_state(), self.get_state(), np.array([[1, 1] for _ in range(self.n_agents)]).astype(np.int32)
    
    def seed(self, seed):
        np.random.seed(seed)
    
    def render(self):
        pass
    
    def close(self):
        pass