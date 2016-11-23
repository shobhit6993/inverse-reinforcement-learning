from enum import Enum

# Number of slots to be filled.
NUM_SLOTS = 3

# Controls the fraction of total confirmations that are explicit.
AGENT_CONFIRM_PROBABILITY = 0.8



class UserStateStatus(Enum):
    EMPTY = "empty"
    CONFIRMED = "confirmed"
    PROVIDED = "provided"


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
