import copy
import numpy as np
import math
import torch
import torch.nn as nn
import yaml
import os
import os.path as osp
from pathlib import Path
from eir_mappo.env.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv
import json
from tensorboardX import SummaryWriter
import random


def _t2n(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def _flatten(T, N, x):
    return x.reshape(T * N, *x.shape[2:])


def _sa_cast(x):
    return x.transpose(1, 0, 2).reshape(-1, *x.shape[2:])

def _ma_cast(x):
    return x.transpose(1, 2, 0, 3).reshape(-1, *x.shape[3:])


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def check(input):
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output


def get_grad_norm(it):
    sum_grad = 0
    for x in it:
        if x.grad is None:
            continue
        sum_grad += x.grad.norm() ** 2
    return math.sqrt(sum_grad)


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * ( (epoch - 1) / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def softmax(x):
    x = x.copy()
    x -= np.max(x, axis=-1, keepdims=True)
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (e > d).float()
    return a*e**2/2 + b*d*(abs(e)-d/2)


def mse_loss(e):
    return e**2/2


def get_shape_from_obs_space(obs_space):
    if obs_space.__class__.__name__ == 'Box':
        obs_shape = list(obs_space.shape)
    elif obs_space.__class__.__name__ == 'list':
        obs_shape = obs_space
    else:
        raise NotImplementedError
    return obs_shape

def get_obs_space_type(obs_space):
    return obs_space.__class__.__name__


def get_shape_from_act_space(act_space):
    if act_space.__class__.__name__ == 'Discrete':
        act_shape = 1
    elif act_space.__class__.__name__ == "Box":
        act_shape = act_space.shape[0]
    return act_shape


def make_train_env(env_name, seed, n_threads, env_args):
    """Make env for training."""
    if env_name == "dexhands":
        from eir_mappo.env.dexhands.dexhands_env import DexHandsEnv
        return DexHandsEnv(
            {"n_threads": n_threads, **env_args})

    def get_env_fn(rank):
        def init_env():
            if env_name == "smac" or env_name == "smac_traitor":
                from eir_mappo.env.smac.StarCraft2_Env import StarCraft2Env
                env = StarCraft2Env(env_args)
            elif env_name == "smacv2":
                from eir_mappo.env.smacv2.smacv2_env import SMACv2Env
                env = SMACv2Env(env_args)
            elif env_name == "mamujoco":
                from eir_mappo.env.mamujoco.multiagent_mujoco.mujoco_multi import MujocoMulti
                env = MujocoMulti(env_args=env_args)
            elif env_name == "pettingzoo_mpe":
                from eir_mappo.env.pettingzoo_mpe.pettingzoo_mpe_env import PettingZooMPEEnv
                assert env_args["scenario"] in ["simple_v2", "simple_spread_v2", "simple_reference_v2",
                                                "simple_speaker_listener_v3", "simple_spread_custom"], "only cooperative scenarios in MPE are supported"
                env = PettingZooMPEEnv(env_args)
            elif env_name == "gym":
                from eir_mappo.env.gym.gym_env import GYMEnv
                env = GYMEnv(env_args)
            elif env_name == "football":
                from eir_mappo.env.football.football_env import FootballEnv
                env = FootballEnv(env_args)
            elif env_name == "toy":
                from eir_mappo.env.toy_example.toy_example import ToyExample
                env = ToyExample(env_args)
            elif env_name == "lbforaging":
                from eir_mappo.env.lbforaging import ForagingEnv
                env = ForagingEnv(env_args)
            elif env_name == "rware":
                from eir_mappo.env.rware import Warehouse
                env = Warehouse(env_args)
            elif env_name == "rendezvous":
                from eir_mappo.env.ma_envs.rendezvous_env import RENDEEnv
                env = RENDEEnv(env_args)
            elif env_name == "pursuit":
                from eir_mappo.env.ma_envs.pursuit_env import PursuitEnv
                env = PursuitEnv(env_args)
            elif env_name == "navigation":
                from eir_mappo.env.ma_envs.navigation_env import NavigationEnv
                env = NavigationEnv(env_args)
            elif env_name == "cover":
                from eir_mappo.env.ma_envs.cover_env import CoverEnv
                env = CoverEnv(env_args)
            else:
                print("Can not support the " +
                      env_name + "environment.")
                raise NotImplementedError
            env.seed(seed + rank * 1000)
            return env

        return init_env

    if n_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(n_threads)])


def make_eval_env(env_name, seed, n_threads, env_args):
    """Make env for evaluation."""
    if env_name == "dexhands":
        raise NotImplementedError

    def get_env_fn(rank):
        def init_env():
            if env_name == "smac" or env_name == "smac_traitor":
                from eir_mappo.env.smac.StarCraft2_Env import StarCraft2Env
                env = StarCraft2Env(env_args)
            elif env_name == "smacv2":
                from eir_mappo.env.smacv2.smacv2_env import SMACv2Env
                env = SMACv2Env(env_args)
            elif env_name == "mamujoco":
                from eir_mappo.env.mamujoco.multiagent_mujoco.mujoco_multi import MujocoMulti
                env = MujocoMulti(env_args=env_args)
            elif env_name == "pettingzoo_mpe":
                from eir_mappo.env.pettingzoo_mpe.pettingzoo_mpe_env import PettingZooMPEEnv
                env = PettingZooMPEEnv(env_args)
            elif env_name == "gym":
                from eir_mappo.env.gym.gym_env import GYMEnv
                env = GYMEnv(env_args)
            elif env_name == "football":
                from eir_mappo.env.football.football_env import FootballEnv
                env = FootballEnv(env_args)
            elif env_name == "toy":
                from eir_mappo.env.toy_example.toy_example import ToyExample
                env = ToyExample(env_args)
            elif env_name == "lbforaging":
                from eir_mappo.env.lbforaging import ForagingEnv
                env = ForagingEnv(env_args)
            elif env_name == "rware":
                from eir_mappo.env.rware import Warehouse
                env = Warehouse(env_args)
            elif env_name == "rendezvous":
                from eir_mappo.env.ma_envs.rendezvous_env import RENDEEnv
                env = RENDEEnv(env_args)
            elif env_name == "pursuit":
                from eir_mappo.env.ma_envs.pursuit_env import PursuitEnv
                env = PursuitEnv(env_args)
            elif env_name == "navigation":
                from eir_mappo.env.ma_envs.navigation_env import NavigationEnv
                env = NavigationEnv(env_args)
            elif env_name == "cover":
                from eir_mappo.env.ma_envs.cover_env import CoverEnv
                env = CoverEnv(env_args)
            else:
                print("Can not support the " +
                      env_name + "environment.")
                raise NotImplementedError
            env.seed(seed * 50000 + rank * 10000)
            return env

        return init_env

    if n_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(n_threads)])

def make_render_env(env_name, seed, env_args):
    manual_render = True
    manual_expand_dims = True
    manual_delay = True
    env_num = 1
    if env_name == "smac" or env_name == "smac_traitor":
        from eir_mappo.env.smac.StarCraft2_Env import StarCraft2Env
        env = StarCraft2Env(args=env_args)
        env.seed(seed * 60000)
        manual_delay = False
    elif env_name == "smacv2":
        from eir_mappo.env.smacv2.smacv2_env import SMACv2Env
        env = SMACv2Env(args=env_args)
        env.seed(seed * 60000)
        manual_delay = False
    elif env_name == "mamujoco":
        from eir_mappo.env.mamujoco.multiagent_mujoco.mujoco_multi import MujocoMulti
        env = MujocoMulti(env_args=env_args)
        env.seed(seed * 60000)
    elif env_name == "pettingzoo_mpe":
        from eir_mappo.env.pettingzoo_mpe.pettingzoo_mpe_env import PettingZooMPEEnv
        env = PettingZooMPEEnv({**env_args, "render_mode": "human"})
        env.seed(seed * 60000)
    elif env_name == "gym":
        from eir_mappo.env.gym.gym_env import GYMEnv
        env = GYMEnv(env_args)
        env.seed(seed * 60000)
    elif env_name == "football":
        from eir_mappo.env.football.football_env import FootballEnv
        env = FootballEnv(env_args)
        manual_render = False
        env.seed(seed * 60000)
    elif env_name == "dexhands":
        from eir_mappo.env.dexhands.dexhands_env import DexHandsEnv
        env = DexHandsEnv({"n_threads": 64, **env_args})
        manual_render = False
        manual_expand_dims = False
        manual_delay = False
        env_num = 64
    elif env_name == "lbforaging":
        from eir_mappo.env.lbforaging.environment import ForagingEnv
        env = ForagingEnv(env_args)
        env.seed(seed * 60000)
    elif env_name == "toy":
        from eir_mappo.env.toy_example.toy_example import ToyExample
        env = ToyExample(env_args)
        env.seed(seed * 60000)
        manual_delay = False
    elif env_name == "rendezvous":
            # from eir_mappo.envs.rendezvous.rendezvous import RENDEEnv
        from eir_mappo.env.ma_envs.rendezvous_env import RENDEEnv
        env = RENDEEnv(env_args)
        env.seed(seed * 60000)
    elif env_name == "pursuit":
        from eir_mappo.env.ma_envs.pursuit_env import PursuitEnv
        env = PursuitEnv(env_args)
        env.seed(seed * 60000)
    elif env_name == "navigation":
        from eir_mappo.envs.ma_envs.navigation_env import NavigationEnv
        env = NavigationEnv(env_args)
        env.seed(seed * 60000)
    elif env_name == "cover":
        from eir_mappo.envs.ma_envs.cover_env import CoverEnv
        env = CoverEnv(env_args)
        env.seed(seed * 60000)
    else:
        print("Can not support the " +
                env_name + "environment.")
        raise NotImplementedError
    return env, manual_render, manual_expand_dims, manual_delay, env_num

def seed(args):
    """Seed the program."""
    if not args["seed_specify"]:
        args["seed"] = np.random.randint(1000, 10000)
    random.seed(args["seed"])
    np.random.seed(args["seed"])
    os.environ['PYTHONHASHSEED'] = str(args["seed"])
    torch.manual_seed(args["seed"])
    torch.cuda.manual_seed(args["seed"])
    torch.cuda.manual_seed_all(args["seed"])


def init_device(args):
    """Init device."""
    if args["cuda"] and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        if args["cuda_deterministic"]:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
    torch.set_num_threads(args["torch_threads"])
    return device


def init_dir(env, env_args, algo, exp_name, seed):
    """Init directory for saving results."""
    # TODO: move the results directory to the example directory
    if env == "smac" or env == "smac_traitor":
        run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
            0] + "/results") / env / env_args["map_name"] / algo / exp_name / str(seed)
    elif env == "smacv2":
        run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
            0] + "/results") / env / env_args["map_name"] / algo / exp_name / str(seed)
    elif env == "mamujoco":
        run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
            0] + "/results") / env / env_args["scenario"] / env_args["agent_conf"] / str(env_args["agent_obsk"]) / algo / exp_name / str(seed)
    elif env == "pettingzoo_mpe":
        run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
            0] + "/results") / env / env_args["scenario"] / algo / exp_name / str(seed)
    elif env == "gym":
        run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
            0] + "/results") / env / env_args["scenario"] / algo / exp_name / str(seed)
    elif env == "football":
        run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
            0] + "/results") / env / env_args["env_name"] / algo / exp_name / str(seed)
    elif env == "dexhands":
        run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
            0] + "/results") / env / env_args["task"] / algo / exp_name / str(seed)
    elif env in ["toy", "rendezvous", "pursuit", "navigation", "cover"]:
        run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
            0] + "/results") / env / algo / exp_name / str(seed)
    elif env == "lbforaging":
        run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
            0] + "/results") / env / "{}x{}-{}p-{}f".format(env_args["field_size"], env_args["field_size"], env_args["players"], env_args["max_food"]) / algo / exp_name / str(seed)
    elif env == "rware":
        run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
            0] + "/results") / env / "{}-{}p-{}f".format(env_args["size"], env_args["n_agents"], env_args["difficulty"]) / algo / exp_name / str(seed)
    if not run_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = run_dir / curr_run
    os.makedirs(str(run_dir), exist_ok=True)
    log_dir = str(run_dir / 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    writter = SummaryWriter(log_dir)
    save_dir = str(run_dir / 'models')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    return run_dir, log_dir, save_dir, writter


def save_config(args, algo_args, env_args, run_dir):
    """Save the configuration of the program."""
    main_args_dict = args.__dict__
    all_args = {"main_args": main_args_dict,
                "algo_args": algo_args,
                "env_args": env_args}
    config_dir = run_dir / 'config.json'
    with open(config_dir, 'w') as f:
        json.dump(all_args, f)


def get_active_func(activation_func):
    if activation_func == "sigmoid":
        return nn.Sigmoid()
    elif activation_func == "tanh":
        return nn.Tanh()
    elif activation_func == "relu":
        return nn.ReLU()
    elif activation_func == "leaky_relu":
        return nn.LeakyReLU()
    elif activation_func == "selu":
        return nn.SELU()
    elif activation_func == "hardswish":
        return nn.Hardswish()
    elif activation_func == "identity":
        return nn.Identity()
    else:
        assert False, "activation function not supported!"


def get_init_method(initialization_method):
    return nn.init.__dict__[initialization_method]


def get_num_agents(env, env_args, envs):
    """Get the number of agents in the environment."""
    if env == "smac" or env == "smac_traitor":
        from eir_mappo.env.smac.smac_maps import get_map_params
        return get_map_params(env_args["map_name"])["n_agents"]
    elif env == "smacv2":
        return envs.n_agents
    elif env == "mamujoco":
        return envs.n_agents
    elif env == "pettingzoo_mpe":
        return envs.n_agents
    elif env == "gym":
        return envs.n_agents
    elif env == "football":
        return envs.n_agents
    elif env == "dexhands":
        return envs.n_agents
    elif env == "toy":
        return envs.n_agents
    elif env == "lbforaging":
        return envs.n_agents
    elif env == "rware":
        return envs.n_agents
    elif env == "rendezvous":
        return envs.n_agents
    elif env == "pursuit":
        return envs.n_agents
    elif env == "navigation":
        return envs.n_agents
    elif env == "cover":
        return envs.n_agents
