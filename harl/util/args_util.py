import os
import os.path as osp
from pathlib import Path
import yaml


def get_args(algo, env):
    """Load config file for user-specified algo and env."""
    base_path = osp.split(osp.dirname(osp.abspath(__file__)))[0]
    algo_cfg_path = os.path.join(base_path, "configs", "algo", f"{algo}.yaml")
    env_cfg_path = os.path.join(base_path, "configs", "env", f"{env}.yaml")

    with open(algo_cfg_path, "r", encoding="utf-8") as file:
        algo_args = yaml.load(file, Loader=yaml.FullLoader)
    with open(env_cfg_path, "r", encoding="utf-8") as file:
        env_args = yaml.load(file, Loader=yaml.FullLoader)
    return algo_args, env_args


def update_args(unparsed_args, *args):
    """Update loaded config with unparsed command-line arguments."""
    def process(arg):
        # Process an arg by eval-ing it, so users can specify more
        # than just strings at the command line (eg allows for
        # users to give functions as args).
        try:
            return eval(arg)
        except:
            return arg
    keys = [k[2:] for k in unparsed_args[0::2]]  # remove -- from argument
    values = [process(v) for v in unparsed_args[1::2]]
    unparsed_dict = {k: v for k, v in zip(keys, values)}

    def update_dict(dict1, dict2):
        for k in dict2:
            if type(dict2[k]) is dict:
                update_dict(dict1, dict2[k])
            else:
                if k in dict1:
                    dict2[k] = dict1[k]
    for args_dict in args:
        update_dict(unparsed_dict, args_dict)


def update_args_nni(unparsed_args, *args):
    """Update loaded config with unparsed nni arguments."""
    def process(arg):
        # Process an arg by eval-ing it, so users can specify more
        # than just strings at the command line (eg allows for
        # users to give functions as args).
        try:
            return eval(arg)
        except:
            return arg
    unparsed_dict = {k: process(unparsed_args[k]) for k in unparsed_args}

    def update_dict(dict1, dict2):
        for k in dict2:
            if type(dict2[k]) is dict:
                update_dict(dict1, dict2[k])
            else:
                if k in dict1:
                    dict2[k] = dict1[k]
    for args_dict in args:
        update_dict(unparsed_dict, args_dict)
