# **Byzantine Robust Cooperative Multi-Agent Reinforcement Learning as a Bayesian Game**

This repository is the official implementation of the paper accepted by ICLR 2024: Byzantine Robust Cooperative Multi-Agent Reinforcement Learning as a Bayesian Game. It contains the implementation of the EIR-MAPPO defense method, along with several Multi-Agent Reinforcement Learning (MARL) environments used for evaluating our results, such as Toy, LBF, and SMAC.

This repository is based on [PKU-MARL/HARL](https://github.com/PKU-MARL/HARL).

## How to Run This Code

### Environment Setup

The process of environment setup is the same as in [PKU-MARL/HARL](https://github.com/PKU-MARL/HARL). For details, please refer to the README file in that repository.

### Choose the Training algorithm and Environment

Our codebase supports various algorithms and environments, with default parameters specified in `./eir_mappo/configs`. The training algorithm parameters for EIR-MAPPO are located in `./eir_mappo/configs/alg/mappo_advt_belief.yaml`, and parameters for the attacking algorithm are in `./eir_mappo/configs/alg/mappo_traitor_belief.yaml`.

Environment-specific parameters are stored in `./eir_mappo/configs/env`, with the YAML configuration files listed as follows:

| MARL Environment | YAML Configuration File                |
| ---------------- | -------------------------------------- |
| Toy              | `./eir_mappo/configs/env/toy.yaml`          |
| LBF              | `./eir_mappo/configs/env/lbforaging.yaml`   |
| SMAC (Training)  | `./eir_mappo/configs/env/smac.yaml`         |
| SMAC (Attack)    | `./eir_mappo/configs/env/smac_traitor.yaml` |

### Training the Agents and Saving the models

To train the agents, execute the following command as an example:

```bash
python -u main.py --alg mappo_advt_belief --env smac --exp_name train --map_name 4m_vs_3m --seed 1
```

* `--alg`: Sets the algorithm. Using `--alg mappo_advt_belief` indicates the use of the training algorithm EIR-MAPPO, with default parameters stored in `./eir_mappo/configs/alg/mappo_advt_belief.yaml`.
* `--env`: Sets the MARL environment. Specifying `--env smac` selects the SMAC environment for training, with its default parameters located in `./eir_mappo/configs/env/smac.yaml`.
* `--map_name`: Specifies the map to be used for training. If `--map` is not explicitly set, the default map name specified in the environment's configuration file is used. The `map_name` parameter is ignored when the selected environment is `Toy`.
* `--exp_name`:  Names the experiment. If `--exp_name` is not provided, it defaults to `test`.
* `--seed`:  Specifies the seed for initializing the experiment, with its default set in the algorithm's configuration file.

The models and training data are saved in the the following directories:

```bash
# For MARL environments other than Toy
models: ./eir_mappo/results/{env}/{map_name}/mappo_advt_belief/{exp_name}/{seed}/run{iter}/models
data: ./eir_mappo/results/{env}/{map_name}/mappo_advt_belief/{exp_name}/{seed}/run{iter}/logs

# For the Toy MARL environment
models: ./eir_mappo/results/{env}/mappo_advt_belief/{exp_name}/{seed}/run{iter}/models
data: ./eir_mappo/results/{env}/mappo_advt_belief/{exp_name}/{seed}/run{iter}/logs
```

The `{iter}` placeholder in the path is incremented by one with each new run, ensuring that experimental data for the same configuration do not overlap, starting with an initial value of 1.

### Attacking the Models and Saving the Adversarial Policy

To attack the models and train the adversarial agents, execute the following command as an example:

```bash
python -u main.py --alg mappo_traitor_belief --env smac --exp_name attack_eir_mappo --map_name 4m_vs_3m --seed 1 --agent_adversary 0 --model_dir ./eir_mappo/results/smac/4m_vs_3m/mappo_advt_belief/eir_mappo/1/run1/models 
```

* `--alg --env --exp_name --map_name --seed`: Parameters are as previously described.
* `--model_dir`: Specifies the directory containing the model to be attacked.
* `--agent_adversary`: Indicates the index of the adversarial agent within the training environment.

The adversarial agent's index can be configured in the algorithm's configuration file (`./eir_mappo/configs/alg/mappo_traitor_belief.yaml`).

The models and attack data are saved in the following directories:

```bash
# For MARL environments other than Toy
models: ./eir_mappo/results/{env}/{map_name}/mappo_traitor_belief/{exp_name}/{seed}/run{iter}/models
datas: ./eir_mappo/results/{env}/{map_name}/mappo_traitor_belief/{exp_name}/{seed}/run{iter}/logs

# For MARL environments other than Toy
models: ./eir_mappo/results/{env}/mappo_traitor_belief/{exp_name}/{seed}/run{iter}/models
datas: ./eir_mappo/results/{env}/mappo_traitor_belief/{exp_name}/{seed}/run{iter}/logs
```

* `{env}, {map_name}, {exp_name}, {seed}, {iter}`: These placeholders are as previously described.

## Demo Videos

We evaluate our performance under the most arduous non-oblivious attack, where an adversary can manipulate any agent in cooperative tasks and execute an arbitrary learned worst-case policy. We also record the behaviors of the agents under the attack in the videos. These videos showcase our methods alongside the baseline methods in the *12x12-4p-3f-c* configuration of the LBF environment and the *4m vs 3m* scenario of the SMAC MARL environment, as illustrated in the table below.

| Training algorithm | Video Directory                                              |
| ------------------ | ------------------------------------------------------------ |
| MADDPG             | [LBF video](./video/LBF/MADDPG.m4v) [SMAC video](./video/SMAC/MADDPG.m4v) |
| M3DDPG             | [LBF video](./video/LBF/M3DDPG.m4v) [SMAC video](./video/SMAC/M3DDPG.m4v) |
| MAPPO              | [LBF video](./video/LBF/MAPPO.m4v) [SMAC video](./video/SMAC/MAPPO.m4v) |
| RMAAC              | [LBF video](./video/LBF/RMAAC.m4v) [SMAC video](./video/SMAC/RMAAC.m4v) |
| EAR-MAPPO          | [LBF video](./video/LBF/EAR-MAPPO.m4v) [SMAC video](./video/SMAC/EAR-MAPPO.m4v) |
| EIR-MAPPO          | [LBF video](./video/LBF/EIR-MAPPO.m4v) [SMAC video](./video/SMAC/EIR-MAPPO.m4v) |
| True Type          | [LBF video](./video/LBF/True-Type.m4v) [SMAC video](./video/SMAC/True-Type.m4v) |
