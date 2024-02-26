from harl.common.v_critic import VCritic
from harl.common.continuous_q_critic import ContinuousQCritic
from harl.common.twin_continuous_q_critic import TwinContinuousQCritic
from harl.common.discrete_q_critic import DiscreteQCritic
from harl.common.continuous_q_critic_ns import ContinuousQCriticNS

CRITIC_REGISTRY = {
    "happo": VCritic,
    "hatrpo": VCritic,
    "haa2c": VCritic,
    "mappo": VCritic,
    "haddpg": ContinuousQCritic,
    "hatd3": TwinContinuousQCritic,
    "had3qn": DiscreteQCritic,
    "maddpg": ContinuousQCritic,
    "m3ddpg": ContinuousQCriticNS,
}
