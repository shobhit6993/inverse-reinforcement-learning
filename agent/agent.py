"""Handcrafted dialog policy for the dialog agent."""

from numpy.random import binomial

from agent_action import AgentActions
from agent_state import AgentState
from utils.params import AGENT_EXPLICIT_VS_IMPLICIT_CONFIRMATION_PROBABILITY
from utils.params import AgentActionType, AgentStateStatus, UserActionType
from utils.params import NUM_SLOTS


class Agent(object):
    """Class for the agent in a dialog system. It keeps track of the agent's
    state, and picks actions in response to user's actions based on a
    handcoded policy.

    Attributes:
        prev_agent_act (AgentAction): Agent's action at the last timestep.
        state (AgentState): Agent's current state.
    """

    def __init__(self):
        self.state = AgentState()
        self.prev_agent_act = None

    def start_dialog(self):
        """Kicks off the dialog session by having the agent take the first
        action.

        Returns:
            AgentAction: The first action taken by the agent.
        """
        # Agent starts with a GREETING
        self.prev_agent_act = AgentActions.greet.value
        # print("Agent-- [State] " + str(self.state))
        print("Agent-- (Action) " + str(self.prev_agent_act))
        # print("A:" + self.prev_agent_act.type.value)
        return self.prev_agent_act

    def take_turn(self, user_act):
        """Executes an agent turn based on the user's most recent action.

        Args:
            user_act (UserAction): User's most recent action.

        Returns:
            AgentAction: Agent's next action.
        """
        next_action = self.update_state_and_next_action(user_act)
        self.prev_agent_act = next_action
        print("Agent-- [State] " + str(self.state))
        print("Agent-- (Action) " + str(next_action))
        # print("A:" + next_action.type.value)
        return next_action

    def update_state_and_next_action(self, user_act):
        """Updates the agent's state and returns the next action for the agent,
        given the dialog history and the current user-action.

        Args:
            user_act (UserAction): Current user-action.

        Returns:
            AgentAction: Next action to be taken by the agent.

        Raises:
            ValueError: Invalid user action.
        """

        if user_act.type is UserActionType.SILENT:
            return self._handle_silence(user_act)
        elif user_act.type is UserActionType.ONE_SLOT:
            return self._handle_one_slot(user_act)
        elif user_act.type is UserActionType.ALL_SLOTS:
            return self._handle_all_slots(user_act)
        elif user_act.type is UserActionType.CONFIRM:
            return self._handle_confirmation(user_act)
        elif user_act.type is UserActionType.NEGATE:
            return self._handle_negation(user_act)
        elif user_act.type is UserActionType.CLOSE:
            return self._handle_close(user_act)
        else:
            raise ValueError("Invalid user-act {}".format(user_act))

    def _handle_silence(self, user_act):
        """Returns an appropriate agent action after user silence.

        Args:
            user_act (UserAction): User's most recent action.

        Returns:
            AgentAction: The action to be taken.
        """
        assert user_act.type is UserActionType.SILENT, (
            "Invalid user action: {} in method: {}"
            .format(user_act, self._handle_silence.__name__))

        # On user silence, if the previous agent action was to greet, then ask
        # for a slot. If the user wants to terminate the session, the agent
        # obliges. Otherwise, just repeat the previous action.

        if self.prev_agent_act.type is AgentActionType.GREET:
            return self._ask_confirm_or_close()
        elif self.prev_agent_act.type is AgentActionType.CLOSE:
            return AgentActions.close.value
        else:
            return self.prev_agent_act

    def _handle_one_slot(self, user_act):
        """Returns an appropriate agent action after user provides a slot.

        Args:
            user_act (UserAction): User's most recent action.

        Returns:
            AgentAction: The action to be taken.

        Raises:
            ValueError: Invalid previous agent action.
        """
        assert user_act.type is UserActionType.ONE_SLOT, (
            "Invalid user action: {} in method: {}"
            .format(user_act, self._handle_one_slot.__name__))

        # On being provided with one slot by the user, the agent marks
        # it "OBTAINED" and goes for confirmation if its last action was a
        # greeting. If agent's requested for a slot, then the requested slot is
        # marked "OBTAINED", not the provided slot. If the agent requested for
        # an implicit confirmation, then the slot that's meant to be confirmed
        # is marked "CONFIRMED", and the slot requested -- potentially
        # different from the one provided -- is marked "OBTAINED"; the agent,
        # then, goes for a confirmation. If the user wants to terminate the
        # session, the agent obliges. In all the
        # other cases, the agent repeats its previous action.

        if self.prev_agent_act.type is AgentActionType.GREET:
            self.state.mark_slot_as_obtained(user_act.slot_id, True)
            return self._confirm()
        elif self.prev_agent_act.type is AgentActionType.ASK_SLOT:
            # Mark the slot requested by agent -- potentially different from
            # the one provided by the user -- as "OBTAINED".
            self.state.mark_slot_as_obtained(self.prev_agent_act.ask_id, True)
            return self._confirm()
        elif self.prev_agent_act.type is AgentActionType.CONFIRM_ASK:
            # Mark the slot for which implicit confirmation was requested
            # as "CONFIRMED".
            self.state.mark_slot_as_confirmed(self.prev_agent_act.confirm_id)
            # Update the slot for which the agent requested information.
            # Note that this might not be the slot user provided.
            self.state.mark_slot_as_obtained(self.prev_agent_act.ask_id)
            return self._confirm()
        elif self.prev_agent_act.type is AgentActionType.CLOSE:
            return AgentActions.close.value
        elif self.prev_agent_act.type is AgentActionType.EXPLICIT_CONFIRM:
            return self.prev_agent_act
        else:
            raise ValueError("Invalid previous agent act {} when user-act "
                             "is {}".format(self.prev_agent_act.type.value,
                                            user_act.type.value))

    def _handle_all_slots(self, user_act):
        """Returns an appropriate agent action after user provides all slots.

        Args:
            user_act (UserAction): User's most recent action.

        Returns:
            AgentAction: The action to be taken.

        Raises:
            ValueError: Invalid previous agent action.
        """
        assert user_act.type is UserActionType.ALL_SLOTS, (
            "Invalid user action: {} in method: {}"
            .format(user_act, self._handle_all_slots.__name__))

        # On being provided with all slots by the user, the agent marks
        # all of them "OBTAINED" and goes for confirmation if its last action
        # was a greeting. If the user wants to terminate the session, the agent
        # obliges. In all other cases, the agent repeats its previous action.

        if self.prev_agent_act.type is AgentActionType.GREET:
            self._mark_all_slots_as_obtained()
            return self._ask_confirm_or_close()
        elif self.prev_agent_act.type is AgentActionType.CLOSE:
            return AgentActions.close.value
        elif (self.prev_agent_act.type is AgentActionType.EXPLICIT_CONFIRM or
              self.prev_agent_act.type is AgentActionType.ASK_SLOT or
              self.prev_agent_act.type is AgentActionType.CONFIRM_ASK):
            return self.prev_agent_act
        else:
            raise ValueError("Invalid previous agent act {} when user-act "
                             "is {}".format(self.prev_agent_act.type.value,
                                            user_act.type.value))

    def _handle_confirmation(self, user_act):
        """Returns an appropriate agent action after user confirms a slot.

        Args:
            user_act (UserAction): User's most recent action.

        Returns:
            AgentAction: The action to be taken.

        Raises:
            ValueError: Invalid previous agent action.
        """
        assert user_act.type is UserActionType.CONFIRM, (
            "Invalid user action: {} in method: {}"
            .format(user_act, self._handle_confirmation.__name__))

        # On being provided with a confirmation for a slot by the user, the
        # agent marks that slot "CONFIRMED" if it asked for an explicit or an
        # implicit confirmation for that slot. If the user wants to terminate
        # the session, the agent obliges. In all other cases, the agent repeats
        # its previous action.

        if (self.prev_agent_act.type is AgentActionType.EXPLICIT_CONFIRM or
                self.prev_agent_act.type is AgentActionType.CONFIRM_ASK):
            # If the user confirms the same slot as the one whose confirmation
            # was sought either explicitly or implicitly, mark it as
            # "CONFIRMED". Otherwise, repeat.
            if self.prev_agent_act.confirm_id == user_act.slot_id:
                self.state.mark_slot_as_confirmed(user_act.slot_id)
                return self._ask_confirm_or_close()
            else:
                return self.prev_agent_act
        elif self.prev_agent_act.type is AgentActionType.CLOSE:
            return AgentActions.close.value
        elif (self.prev_agent_act.type is AgentActionType.GREET or
              self.prev_agent_act.type is AgentActionType.ASK_SLOT):
            return self.prev_agent_act
        else:
            raise ValueError("Invalid previous agent act {} when user-act "
                             "is {}".format(self.prev_agent_act.type.value,
                                            user_act.type.value))

    def _handle_negation(self, user_act):
        """Returns an appropriate agent action after user negates a slot.

        Args:
            user_act (UserAction): User's most recent action.

        Returns:
            AgentAction: The action to be taken.

        Raises:
            ValueError: Invalid previous agent action.
        """
        assert user_act.type is UserActionType.NEGATE, (
            "Invalid user action: {} in method: {}"
            .format(user_act, self._handle_negation.__name__))

        # On being provided with a negation for a slot by the user, the
        # agent marks that slot "EMPTY" if it asked for an explicit or an
        # implicit negation for that slot. If the user wants to terminate
        # the session, the agent obliges. In all other cases, the agent repeats
        # its previous action.

        if (self.prev_agent_act.type is AgentActionType.EXPLICIT_CONFIRM or
                self.prev_agent_act.type is AgentActionType.CONFIRM_ASK):
            # If the user negates the same slot as the one whose confirmation
            # was sought either explicitly or implicitly, mark it as
            # "EMPTY". Otherwise, repeat.
            if self.prev_agent_act.confirm_id == user_act.slot_id:
                self.state.mark_slot_as_empty(user_act.slot_id)
                return self._ask_confirm_or_close()
            else:
                return self.prev_agent_act
        elif self.prev_agent_act.type is AgentActionType.CLOSE:
            return AgentActions.close.value
        elif (self.prev_agent_act.type is AgentActionType.GREET or
              self.prev_agent_act.type is AgentActionType.ASK_SLOT):
            return self.prev_agent_act
        else:
            raise ValueError("Invalid previous agent act {} when user-act "
                             "is {}".format(self.prev_agent_act.type.value,
                                            user_act.type.value))

    def _handle_close(self, user_act):
        """Returns an appropriate agent action after user wishes to terminate.

        Args:
            user_act (UserAction): User's most recent action.

        Returns:
            AgentAction: The action to be taken.
        """
        assert user_act.type is UserActionType.CLOSE, (
            "Invalid user action: {} in method: {}"
            .format(user_act, self._handle_close.__name__))

        # If the user wants to terminate the session, the agent always
        # obliges.

        return AgentActions.close.value

    def _mark_all_slots_as_obtained(self):
        """Marks all slots "OBTAINED"
        """
        for id_ in xrange(NUM_SLOTS):
            if self.state.slots[id_] is AgentStateStatus.EMPTY:
                self.state.mark_slot_as_obtained(id_)

    def _ask_or_close(self):
        """If there is an EMPTY slot, retuns an action to request that slot.
        Otherwise, returns an action for terminating the dialog session.

        Returns:
            AgentAction: A request-slot action, or one to terminate the dialog
                session, whichever is appropriate.
        """
        slot_id = self.state.get_empty_slot()
        if slot_id is None:
            return AgentActions.close.value
        else:
            return AgentActions.ask_slot.value[slot_id]

    def _ask_confirm_or_close(self):
        """If there is an unconfirmed slot, returns an action to confirm it.
        Otherwise, returns an action to request an EMPTY slot. If none of the
        above are possible, returns an action to terminate the dialog session.

        An "unconfirmed" slot is one which is marked "OBTAINED", but not yet
        "CONFIRMED".

        Returns:
            AgentAction: A request-slot action, or a confirm-slot action, or
            one to terminate the dialog session, whichever is appropriate.
        """
        empty_slot_id = self.state.get_empty_slot()
        if empty_slot_id is None:
            unconfirmed_slot_id = self.state.get_unconfirmed_slot()
            if unconfirmed_slot_id is None:
                return AgentActions.close.value
            else:
                return AgentActions.explicit_confirm.value[unconfirmed_slot_id]
        else:
            return AgentActions.ask_slot.value[empty_slot_id]

    def _confirm(self):
        """Returns an action to confirm an unconfirmed slot. The confirmation
        could either be explicit or implicit. In the latter case, the action
        requests an EMPTY slot in addition to implicitly confirming an
        unconfirmed one.

        An "unconfirmed" slot is one which is marked "OBTAINED", but not yet
        "CONFIRMED".

        Returns:
            AgentAction: An action to confirm a slot.
        """
        # Controls the fraction of total confirmations that are explicit.
        b = binomial(1, AGENT_EXPLICIT_VS_IMPLICIT_CONFIRMATION_PROBABILITY)
        if b == 1:
            return self._explicit_confirm()
        else:
            return self._implicit_confirm()

    def _explicit_confirm(self):
        """Returns an action to explicitly confirm an unconfirmed slot.
        If there is no slot that can be confirmed, then it invokes the
        `_ask_confirm_or_close` method to return an appropriate action.

        An "unconfirmed" slot is one which is marked "OBTAINED", but not yet
        "CONFIRMED".

        Returns:
            AgentAction: An action to explicitly confirm a slot, or another
                appropriate action if no such slot is available.
        """
        unconfirmed_slot_id = self.state.get_unconfirmed_slot()
        if unconfirmed_slot_id is not None:
            return AgentActions.explicit_confirm.value[unconfirmed_slot_id]
        else:
            return self._ask_confirm_or_close()

    def _implicit_confirm(self):
        """Returns an action to implicitly confirm an unconfirmed slot. The
        `_explicit_confirm` method is invoked if there is not EMPTY slot.

        An "unconfirmed" slot is one which is marked "OBTAINED", but not yet
        "CONFIRMED".

        Returns:
            AgentAction: An action to implicitly confirm a slot, if possible.
        """
        empty_slot_id = self.state.get_empty_slot()
        unconfirmed_slot_id = self.state.get_unconfirmed_slot()

        # If there is no EMPTY slot, implicit confirmation isn't possible
        # because there isn't a slot left to request alongside the
        # implicit confirmation. In such a case, leave it upto the
        # `_explicit_confirm` method to pick an appropriate action.
        if empty_slot_id is None:
            return self._explicit_confirm()
        else:
            return (AgentActions.confirm_ask
                    .value[unconfirmed_slot_id][empty_slot_id])
