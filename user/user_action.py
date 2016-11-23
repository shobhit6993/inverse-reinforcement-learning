from enum import Enum
from utils.params import NUM_SLOTS, UserActionType


class UserAction:
    def __init__(self, type_, slot_id):
        self.type = type_
        self.slot_id = slot_id


class UserActions(Enum):
    silent = UserActionType.SILENT
    all_slots = UserActionType.ALL_SLOTS
    one_slot = [UserAction(UserActionType.ONE_SLOT, id_)
                for id_ in xrange(NUM_SLOTS)]
    confirm = [UserAction(UserActionType.CONFIRM, id_)
               for id_ in xrange(NUM_SLOTS)]
    negate = [UserAction(UserActionType.NEGATE, id_)
              for id_ in xrange(NUM_SLOTS)]
    close_dialog = UserAction(UserAction.CLOSE, None)
