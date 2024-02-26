# **Byzantine Robust Cooperative Multi-Agent Reinforcement Learning as a Bayesian Game**

This repository contains the implementation of the EIR-MAPPO defense method, along with several Multi-Agent Reinforcement Learning (MARL) environments used for evaluating our results, such as Toy, LBF, and SMAC.

## How to Run This Code

### Choose the Training algorithm and Environment

Our codebase supports various algorithms and environments, with default parameters specified in `./harl/configs`. The training algorithm parameters for EIR-MAPPO are located in `./harl/configs/alg/mappo_advt_belief.yaml`, and parameters for the attacking algorithm are in `./harl/configs/alg/mappo_traitor_belief.yaml`.

Environment-specific parameters are stored in `./harl/configs/env`, with the YAML configuration files listed as follows:

| MARL Environment | YAML Configuration File                |
| ---------------- | -------------------------------------- |
| Toy              | `./harl/configs/env/toy.yaml`          |
| LBF              | `./harl/configs/env/lbforaging.yaml`   |
| SMAC (Training)  | `./harl/configs/env/smac.yaml`         |
| SMAC (Attack)    | `./harl/configs/env/smac_traitor.yaml` |

### Training the Agents and Saving the models

To train the agents, execute the following command as an example:

```bash
python -u main.py --alg mappo_advt_belief --env smac --exp_name train --map_name 4m_vs_3m --seed 1
```

* `--alg`: Sets the algorithm. Using `--alg mappo_advt_belief` indicates the use of the training algorithm EIR-MAPPO, with default parameters stored in `./harl/configs/alg/mappo_advt_belief.yaml`.
* `--env`: Sets the MARL environment. Specifying `--env smac` selects the SMAC environment for training, with its default parameters located in `./harl/configs/env/smac.yaml`.
* `--map_name`: Specifies the map to be used for training. If `--map` is not explicitly set, the default map name specified in the environment's configuration file is used. The `map_name` parameter is ignored when the selected environment is `Toy`.
* `--exp_name`:  Names the experiment. If `--exp_name` is not provided, it defaults to `test`.
* `--seed`:  Specifies the seed for initializing the experiment, with its default set in the algorithm's configuration file.

The models and training data are saved in the the following directories:

```bash
# For MARL environments other than Toy
models: ./harl/results/{env}/{map_name}/mappo_advt_belief/{exp_name}/{seed}/run{iter}/models
data: ./harl/results/{env}/{map_name}/mappo_advt_belief/{exp_name}/{seed}/run{iter}/logs

# For the Toy MARL environment
models: ./harl/results/{env}/mappo_advt_belief/{exp_name}/{seed}/run{iter}/models
data: ./harl/results/{env}/mappo_advt_belief/{exp_name}/{seed}/run{iter}/logs
```

The `{iter}` placeholder in the path is incremented by one with each new run, ensuring that experimental data for the same configuration do not overlap, starting with an initial value of 1.

### Attacking the Models and Saving the Adversarial Policy

To attack the models and train the adversarial agents, execute the following command as an example:

```bash
python -u main.py --alg mappo_traitor_belief --env smac --exp_name attack_eir_mappo --map_name 4m_vs_3m --seed 1 --agent_adversary 0 --model_dir ./harl/results/smac/4m_vs_3m/mappo_advt_belief/eir_mappo/1/run1/models 
```

* `--alg --env --exp_name --map_name --seed`: Parameters are as previously described.
* `--model_dir`: Specifies the directory containing the model to be attacked.
* `--agent_adversary`: Indicates the index of the adversarial agent within the training environment.

The adversarial agent's index can be configured in the algorithm's configuration file (`./harl/configs/alg/mappo_traitor_belief.yaml`).

The models and attack data are saved in the following directories:

```bash
# For MARL environments other than Toy
models: ./harl/results/{env}/{map_name}/mappo_traitor_belief/{exp_name}/{seed}/run{iter}/models
datas: ./harl/results/{env}/{map_name}/mappo_traitor_belief/{exp_name}/{seed}/run{iter}/logs

# For MARL environments other than Toy
models: ./harl/results/{env}/mappo_traitor_belief/{exp_name}/{seed}/run{iter}/models
datas: ./harl/results/{env}/mappo_traitor_belief/{exp_name}/{seed}/run{iter}/logs
```

* `{env}, {map_name}, {exp_name}, {seed}, {iter}`: These placeholders are as previously described.