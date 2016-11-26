"""Parameters and global constants used in the codebase."""

from enum import Enum

# Number of dialog sessions to be simulated for building the corpus
NUM_SESSIONS_IN_CORPUS = 100000

# Number of slots to be filled.
NUM_SLOTS = 3

# Controls the fraction of total confirmations that are explicit.
AGENT_EXPLICIT_VS_IMPLICIT_CONFIRMATION_PROBABILITY = 0.8

# Discount factor
GAMMA = 0.95


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
    SILENT = "silent"
    ALL_SLOTS = "provide-all-slots"
    ONE_SLOT = "provide-one-slot"
    CONFIRM = "confirm"
    NEGATE = "negate"
    CLOSE = "close"


class AgentActionType(Enum):
    GREET = "greet"
    ASK_SLOT = "ask_slot"
    EXPLICIT_CONFIRM = "explicit_confirm"
    CONFIRM_ASK = "confirm_and_ask"
    CLOSE = "close"
