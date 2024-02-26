import socket
from absl import flags
from harl.env.smac.logger import SMACLogger, BaseLogger
from harl.env.smacv2.logger import SMACv2Logger
from harl.env.mamujoco.logger import MAMuJoCoLogger
from harl.env.pettingzoo_mpe.logger import PettingZooMPELogger
from harl.env.gym.logger import GYMLogger
from harl.env.football.logger import FootballLogger
from harl.env.dexhands.logger import DexHandsLogger
from harl.env.toy_example.logger import ToyLogger
from harl.env.ma_envs.rendezvous_logger import RendezvousLogger
from harl.env.ma_envs.pursuit_logger import PursuitLogger
from harl.env.ma_envs.navigation_logger import NavigationLogger
from harl.env.ma_envs.cover_logger import CoverLogger

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
