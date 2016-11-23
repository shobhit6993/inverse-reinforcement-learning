from enum import Enum

from utils.params import AgentActionType
from utils.params import NUM_SLOTS


class AgentAction:
    def __init__(self, type_, ask_id, confirm_id):
        self.type = type_
        self.ask_id = ask_id
        self.confirm_id = confirm_id

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
