""" Dialog agent's actions."""

from enum import Enum

from utils.params import AgentActionType, NUM_SLOTS


class AgentAction(object):
    """Action class for dialog agent. An action is defined by the its type,
    and the identifier for the slot which is being asked and/or confirmed
    by this action.

    Attributes:
        ask_id (int or None): Id of slot being requested by this action.
            Optionally, None.
        confirm_id (int or None): Id of slot being confirmed by this
            action. Optionally, None.
        ask_id (TYPE): Description
        confirm_id (TYPE): Description
        type (AgentActionType): Type of action
    """

    def __init__(self, type_, ask_id, confirm_id):
        self.type = type_
        self.ask_id = ask_id
        self.confirm_id = confirm_id

    def __hash__(self):
        # Only the `type` attribute is used for hashing.
        return hash(self.type)

    def __eq__(self, other):
        # Only the `type` attribute is used to test equality.
        if type(other) is AgentAction:
            return (self.type is other.type)
        else:
            return False

    def __ne__(self, other):
        # Only the `type` attribute is used to test inequality.
        return not (self == other)

    def __str__(self):
        return "Type: {}, Ask_id: {}, Confirm_id: {}".format(
            self.type.value, self.ask_id, self.confirm_id)


class AgentActions(Enum):
    # `AgentAction` for greeting.
    greet = AgentAction(AgentActionType.GREET, None, None)

    # Multiple `AgentAction`s for requesting a slot, one for each slot.
    ask_slot = [AgentAction(AgentActionType.ASK_SLOT, i, None)
                for i in xrange(NUM_SLOTS)]

    # Multiple `AgentAction`s for explicitly confirming a slot, one for each
    # slot.
    explicit_confirm = [AgentAction(AgentActionType.EXPLICIT_CONFIRM, None, i)
                        for i in xrange(NUM_SLOTS)]

    # A matrix of `AgentAction`s corresponding to implicit confirmation and
    # slot request. `AgentAction` in cell i,j performs implicit confirmation
    # for slot# i and asks for information about slot# j.
    confirm_ask = [[AgentAction(AgentActionType.CONFIRM_ASK, ask_id, conf_id)
                    for ask_id in xrange(NUM_SLOTS)]
                   for conf_id in xrange(NUM_SLOTS)]
    close = AgentAction(AgentActionType.CLOSE, None, None)
    bad_close = AgentAction(AgentActionType.BAD_CLOSE, None, None)
    # Delete the loop variables to prevent them from being treated as
    # enum members. Sigh.
    del i, ask_id, conf_id
