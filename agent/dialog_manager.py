"""Handcrafted dialog policy for the dialog manager."""

from utils.params import AgentActionType, UserActionType
from utils.params import AGENT_CONFIRM_PROBABILITY
from agent.agent_action import AgentActions

from numpy.random import binomial


class DialogManager:
    def __init__(self, init_state):
        self.state = init_state
        self.prev_agent_act = None

    def update_state_and_next_action(self, user_act):
        # If the user is silent, repeat previous action, except when the last
        # agent action was "greet", in which case the agent should ask for
        # information about a slot
        if user_act.type is UserActionType.SILENT:
            if self.prev_agent_act.type is AgentActionType.GREET:
                return self._ask_or_close()
            else:
                return self.prev_agent_act

        # If the agent requested a slot, and the user provided one or more
        # slots, then update that slot and move to confirmation.
        if self.prev_agent_act.type is AgentActionType.ASK_SLOT:
            if (user_act.type is UserActionType.ONE_SLOT or
                    user_act.type is UserActionType.ALL_SLOTS):
                self.state.mark_slot_as_provided(self.prev_agent_act.ask_id)
                return self._confirm()
            else:
                raise ValueError("User-act of type {} is not supported when"
                                 "previous agent act was {}"
                                 .format(self.user_act.type.value,
                                         self.prev_agent_act.type.value))

        # If the agent requested an explicit confirmation, then update the slot
        # based on whether the user agrees or dissents. Follow this with a
        # request for another slot, or close dialog, as appropriate.
        if self.prev_agent_act.type is AgentActionType.EXPLICIT_CONFIRM:
            if user_act.type is UserActionType.CONFIRM:
                self.state.mark_slot_as_confirmed(
                    self.prev_agent_act.confirm_id)
                return self._ask_or_close()
            elif user_act.type is UserActionType.NEGATE:
                self.state.mark_slot_as_empty(self.prev_agent_act.confirm_id)
                return self._ask_or_close()
            else:
                raise ValueError("User-act of type {} is not supported when"
                                 "previous agent act was {}"
                                 .format(self.user_act.type.value,
                                         self.prev_agent_act.type.value))

        # Implicit confirmation involves a confirmation for a previously
        # requested slot along with a new request for another slot.
        # If the agent went for an implicit confirmation at the last step, then
        # update the slot being confirmed as per the user-response: Negation
        # should result in the slot being marked EMPTY, while user's act of
        # providing information for the requested slot should be treated as an
        # affirmation for the slot being confirmed.
        if self.prev_agent_act.type is AgentActionType.CONFIRM_ASK:
            if user_act.type is UserActionType.ONE_SLOT:
                # Mark the slot for which implicit confirmation was requested
                # as "CONFIRMED".
                self.state.mark_slot_as_confirmed(
                    self.prev_agent_act.confirm_id)
                # Update the slot for which the user provided information.
                self.state.mark_slot_as_provided(self.prev_agent_act.ask_id)
                return self._confirm()
            elif user_act.type is UserActionType.NEGATE:
                self.state.mark_slot_as_empty(self.prev_agent_act.confirm_id)
                return self._ask_or_close()
            else:
                raise ValueError("User-act of type {} is not supported when"
                                 "previous agent act was {}"
                                 .format(self.user_act.type.value,
                                         self.prev_agent_act.type.value))

        # If the user wants to end the conversation, the agent should oblidge.
        if self.user_act.type is UserActionType.CLOSE:
            return AgentActions.close_dialog.value

    def _get_empty_slot_id(self):
        return self.state.get_empty_slot()

    def _ask_or_close(self):
        slot_id = self._get_empty_slot_id()
        if slot_id is None:
            return AgentActions.close_dialog.value
        else:
            return AgentActions.ask_slot.value[slot_id]

    def _confirm(self):
        b = binomial(1, AGENT_CONFIRM_PROBABILITY)
        if b == 1:
            return self._explicit_confirm()
        else:
            return self._implicit_confirm()

    def _explicit_confirm(self):
        return AgentActions.explicit_confirm.value[self.prev_agent_act.ask_id]

    def _implicit_confirm(self):
        empty_slot_id = self._get_empty_slot_id()
        confirm_slot_id = self.prev_agent_act.ask_id
        if empty_slot_id is None:
            return AgentActions.explicit_confirm.value[confirm_slot_id]
        else:
            return (AgentActions.confirm_ask
                    .value[confirm_slot_id][empty_slot_id])
