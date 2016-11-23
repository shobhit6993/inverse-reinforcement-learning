from enum import Enum

from utils.params import NUM_SLOTS
from utils.params import UserActionType


class UserAction:
    def __init__(self, type_, slot_id):
        self.type = type_
        self.slot_id = slot_id

    def __str__(self):
        return "Type: {}, Slot_id: {}".format(self.type.value, self.slot_id)


class UserActions(Enum):
    silent = UserAction(UserActionType.SILENT, None)
    all_slots = UserAction(UserActionType.ALL_SLOTS, None)
    one_slot = [UserAction(UserActionType.ONE_SLOT, id_)
                for id_ in xrange(NUM_SLOTS)]
    confirm = [UserAction(UserActionType.CONFIRM, id_)
               for id_ in xrange(NUM_SLOTS)]
    negate = [UserAction(UserActionType.NEGATE, id_)
              for id_ in xrange(NUM_SLOTS)]
    close = UserAction(UserActionType.CLOSE, None)
