from enum import Enum

# Number of dialog sessions to be simulated for building the corpus
NUM_SESSIONS_IN_CORPUS = 100000

# Number of slots to be filled.
NUM_SLOTS = 3

# Following a request for a slot from the system, this is the probability with
# which the user provides the requested slot. Alternatively, the user provides
# information for all slots.
USER_ONE_SLOT_VS_ALL_SLOTS_PROBABILITY = 0.95

# If the system greets, this is the probability with which the user stays
# silent. Alternatively, the user provides information for all slots.
USER_SILENT_VS_ALL_SLOTS_PROBABILITY = 0.7

# Following an implicit confirmation from the system, this is the probability
# with which the user provides the requested slot, implicitly confirming the
# slot being confirmed. Alternatively, the user negates the confirmation
# without providing any new information.
USER_ONE_SLOT_VS_NEGATE_PROBABILITY = 0.9

# Controls the fraction of total confirmations that are explicit.
AGENT_EXPLICIT_VS_IMPLICIT_CONFIRMATION_PROBABILITY = 0.8


class UserStateStatus(Enum):
    EMPTY = "empty"
    PROVIDED = "provided"
    CONFIRMED = "confirmed"


class AgentStateStatus(Enum):
    EMPTY = "empty"
    OBTAINED = "obtained"
    CONFIRMED = "confirmed"


class UserActionType(Enum):
    # __order__ = ("SILENT ALL_SLOTS ONE_SLOT CONFIRM NEGATE CLOSE")
    SILENT = "silent"
    ALL_SLOTS = "provide-all-slots"
    ONE_SLOT = "provide-one-slot"
    CONFIRM = "confirm"
    NEGATE = "negate"
    CLOSE = "close"


class AgentActionType(Enum):
    # __order__ = "GREET ASK_SLOT EXPLICIT_CONFIRM CONFIRM_ASK CLOSE"
    GREET = "greet"
    ASK_SLOT = "ask_slot"
    EXPLICIT_CONFIRM = "explicit_confirm"
    CONFIRM_ASK = "confirm_and_ask"
    CLOSE = "close"
