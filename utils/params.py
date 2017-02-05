"""Parameters and global constants used in the codebase."""

from enum import Enum
from numpy.random import randint

# Number of dialog sessions to be run for calculation of feature expectations.
NUM_SESSIONS_FE = 1000

# Number of slots to be filled.
NUM_SLOTS = 3

# Controls the fraction of total confirmations that are explicit.
AGENT_EXPLICIT_VS_IMPLICIT_CONFIRMATION_PROBABILITY = 0.5

# Discount factor
GAMMA = 0.95

# Number of episodes to run for Q-learning
Q_LEARNING_EPISODES = 100

# Learning rate for Q-learning
Q_LEARNING_RATE = 0.1

# Rate of decay for learning rate in Q-learning
Q_DECAY_RATE = 0.9

# Degree of randomness in a poliy.
EPSILON = 0.1

# Rate of decay for degree of randomness in Q-learning policies.
EPSILON_DECAY_RATE = 0.99

# Threshold for IRL
THRESHOLD = 0.001

# File where learnt user simulations are dumped periodically
SIMULATIONS_DUMP_FILE = "./simulations-dump-" + str(randint(1000, 9999))

TAU = 1

# User policy types
class UserPolicyType(Enum):
    handcrafted = 1
    random = 2


class UserStateStatus(Enum):
    EMPTY = "empty"
    PROVIDED = "provided"
    CONFIRMED = "confirmed"


class AgentStateStatus(Enum):
    EMPTY = "empty"
    OBTAINED = "obtained"
    CONFIRMED = "confirmed"


class UserActionType(Enum):
    __order__ = 'SILENT ONE_SLOT ALL_SLOTS CONFIRM NEGATE CLOSE'
    SILENT = "silent"
    ONE_SLOT = "provide-one-slot"
    ALL_SLOTS = "provide-all-slots"
    CONFIRM = "confirm"
    NEGATE = "negate"
    CLOSE = "close"


class AgentActionType(Enum):
    __order__ = 'GREET ASK_SLOT EXPLICIT_CONFIRM CONFIRM_ASK CLOSE BAD_CLOSE'
    GREET = "greet"
    ASK_SLOT = "ask_slot"
    EXPLICIT_CONFIRM = "explicit_confirm"
    CONFIRM_ASK = "confirm_and_ask"
    CLOSE = "close"
    BAD_CLOSE = "bad_close"
