import socket
from absl import flags
from eir_mappo.env.smac.logger import SMACLogger, BaseLogger
from eir_mappo.env.smacv2.logger import SMACv2Logger
from eir_mappo.env.mamujoco.logger import MAMuJoCoLogger
from eir_mappo.env.pettingzoo_mpe.logger import PettingZooMPELogger
from eir_mappo.env.gym.logger import GYMLogger
from eir_mappo.env.football.logger import FootballLogger
from eir_mappo.env.dexhands.logger import DexHandsLogger
from eir_mappo.env.toy_example.logger import ToyLogger
from eir_mappo.env.ma_envs.rendezvous_logger import RendezvousLogger
from eir_mappo.env.ma_envs.pursuit_logger import PursuitLogger
from eir_mappo.env.ma_envs.navigation_logger import NavigationLogger
from eir_mappo.env.ma_envs.cover_logger import CoverLogger

FLAGS = flags.FLAGS
FLAGS(['train_sc.py'])

LOGGER_REGISTRY = {
    "smac": SMACLogger,
    "smac_traitor": SMACLogger,
    "mamujoco": MAMuJoCoLogger,
    "pettingzoo_mpe": PettingZooMPELogger,
    "gym": GYMLogger,
    "football": FootballLogger,
    "dexhands": DexHandsLogger,
    "smacv2": SMACv2Logger,
    "toy": ToyLogger,
    "lbforaging": ToyLogger,
    "rware": ToyLogger,
    "rendezvous": ToyLogger,
    "pursuit": ToyLogger,
    "navigation": ToyLogger,
    "cover": ToyLogger
}
