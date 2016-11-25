"""User's actions."""

from enum import Enum

from utils.params import NUM_SLOTS, UserActionType


class UserAction(object):
    """Action class for user. An action is defined by the its type,
    and the identifier for the slot under consideration.

    Attributes:
        slot_id (TYPE): Id of slot under consideration.
        type (UserActionType): Type of action.
    """

    def __init__(self, type_, slot_id):
        self.type = type_
        self.slot_id = slot_id

    def __str__(self):
        return "Type: {}, Slot_id: {}".format(self.type.value, self.slot_id)


class UserActions(Enum):
    # `UserAction` for staying silent.
    silent = UserAction(UserActionType.SILENT, None)

    # `UserAction` for providing all slots.
    all_slots = UserAction(UserActionType.ALL_SLOTS, None)

    # Multiple `UserAction`s for providing a single slot, one for each slot.
    one_slot = [UserAction(UserActionType.ONE_SLOT, id_)
                for id_ in xrange(NUM_SLOTS)]

    # Multiple `UserAction`s for confirming a single slot, one for each slot.
    confirm = [UserAction(UserActionType.CONFIRM, id_)
               for id_ in xrange(NUM_SLOTS)]

    # Multiple `UserAction`s for negating a single slot, one for each slot.
    negate = [UserAction(UserActionType.NEGATE, id_)
              for id_ in xrange(NUM_SLOTS)]

    # `UserAction` for terminating the dialog session.
    close = UserAction(UserActionType.CLOSE, None)
