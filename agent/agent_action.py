from enum import Enum

from utils.params import AgentActionType
from utils.params import NUM_SLOTS


class AgentAction(object):
    def __init__(self, type_, ask_id, confirm_id):
        self.type = type_
        self.ask_id = ask_id
        self.confirm_id = confirm_id

    def __hash__(self):
        """Returns a hash value for `AgentAction` object.
        Only the `type` attribute is used for hashing.

        Returns:
            int: Hash value of this class object.
        """
        return hash(self.type)

    def __eq__(self, other):
        """Tests if the other `AgentAction` object's value is same as this
        one's. Only the `type` attribute is used to test equality.

        Args:
            other: The object to be compared with.

        Returns:
            Boolean: True if the values of the two objects are same.
        """
        if type(other) is AgentAction:
            return (self.type is other.type)
        else:
            return False

    def __ne__(self, other):
        """Tests if the other `AgentAction` object's value is different from
        this one. Only the `type` attribute is used to test inequality.

        Args:
            other: The object to be compared with.

        Returns:
            Boolean: True if the values of the two objects are different.
        """
        return not (self == other)

    def __str__(self):
        return "Type: {}, Ask_id: {}, Confirm_id: {}".format(
            self.type.value, self.ask_id, self.confirm_id)


class AgentActions(Enum):
    greet = AgentAction(AgentActionType.GREET, None, None)
    ask_slot = [AgentAction(AgentActionType.ASK_SLOT, i, None)
                for i in xrange(NUM_SLOTS)]
    explicit_confirm = [AgentAction(AgentActionType.EXPLICIT_CONFIRM, None, i)
                        for i in xrange(NUM_SLOTS)]

    # A matrix of actions corresponding to implicit confirmation and slot
    # request. Action in cell i,j performs implicit confirmation for slot# i
    # and asks for information about slot# j.
    confirm_ask = [[AgentAction(AgentActionType.CONFIRM_ASK, ask_id, conf_id)
                    for ask_id in xrange(NUM_SLOTS)]
                   for conf_id in xrange(NUM_SLOTS)]
    close = AgentAction(AgentActionType.CLOSE, None, None)

    # Delete the loop variables to prevent them from being treated as
    # enum members. Sigh.
    del i, ask_id, conf_id
