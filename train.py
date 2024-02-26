import argparse
import json
from harl.util.args_util import get_args, update_args
from harl.runner import RUNNER_REGISTRY


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="mappo_advt_belief",
        choices=[
            "mappo_advt_belief",
            "mappo_traitor_belief"
        ],
        help="Algorithm name. Choose from: mappo_advt_belief, mappo_traitor_belief.",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="pettingzoo_mpe",
        choices=[
            "toy",
            "smac",
            "smac_traitor",
            "mamujoco",
            "pettingzoo_mpe",
            "gym",
            "football",
            "dexhands",
            "smacv2",
            "lbforaging",
            "rware",
            "fortattack",
            "wolfpack",
            "rendezvous",
            "pursuit",
            "navigation",
            "cover"
        ],
        help="Environment name. Choose from: toy, smac, smac_traitor, mamujoco, pettingzoo_mpe, gym, football, dexhands, smacv2, lbforaging, rware, fortattack, wolfpack, rendezvous, pursuit, navigation, cover.",
    )
    parser.add_argument(
        "--exp_name", type=str, default="test", help="Experiment name."
    )
    parser.add_argument(
        "--load_config",
        type=str,
        default="",
        help="If set, load existing experiment config file instead of reading from yaml config file.",
    )
    args, unparsed_args = parser.parse_known_args()
    
    if args.load_config != "":
        with open(args.load_config) as f:
            all_config = json.load(f)
        args.algo = all_config["main_args"]["algo"]
        args.env = all_config["main_args"]["env"]
        args.exp_name = all_config["main_args"]["exp_name"]
        algo_args = all_config["algo_args"]
        env_args = all_config["env_args"]
    else:
        algo_args, env_args = get_args(args.algo, args.env)
        update_args(unparsed_args, algo_args, env_args)

    if args.env == "dexhands":
        import isaacgym

    # notes: isaac gym does not support multiple instances, thus cannot eval separately
    if args.env == "dexhands":
        algo_args["eval"]["use_eval"] = False
        algo_args["train"]["episode_length"] = env_args["hands_episode_length"]

    # start training
    runner = RUNNER_REGISTRY[args.algo](args, algo_args, env_args)
    runner.run()
    runner.close()


if __name__ == "__main__":
    main()
