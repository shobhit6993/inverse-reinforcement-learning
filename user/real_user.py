"""Handcrafted dialog policy for the real users."""

from numpy.random import binomial

from utils.params import AgentActionType
from utils.params import NUM_SLOTS
from utils.params import USER_ONE_SLOT_VS_ALL_SLOTS_PROBABILITY
from utils.params import USER_ONE_SLOT_VS_NEGATE_PROBABILITY
from utils.params import USER_SILENT_VS_ALL_SLOTS_PROBABILITY
from utils.params import UserStateStatus
from user_action import UserActions
from user_state import UserState


class RealUser(object):
    def __init__(self):
        self.state = UserState()

    def take_turn(self, system_act):
        """Executes a user turn based on the agent's most recent action.

        Args:
            system_act (AgentAction): Dialog agent's most recent action.

        Returns:
            UserAction: User's next action.
        """
        next_action = self.update_state_and_next_action(system_act)
        # print("User -- [State] " + str(self.state))
        # print("User -- (Action) " + str(next_action))
        print("U:" + next_action.type.value)
        return next_action

    def update_state_and_next_action(self, system_act):
        """Updates the user's state and returns the next action for the user,
        given the dialog history and the current system-action.

        Args:
            system_act (AgentAction): Current agent-action.

        Returns:
            UserAction: Next action to be taken by the user.

        Raises:
            ValueError: Un-supported system-action.
        """

        if system_act.type is AgentActionType.GREET:
            return self._handle_greeting()
        elif system_act.type is AgentActionType.ASK_SLOT:
            return self._handle_slot_request(system_act)
        elif system_act.type is AgentActionType.EXPLICIT_CONFIRM:
            return self._handle_explicit_confirmation(system_act)
        elif system_act.type is AgentActionType.CONFIRM_ASK:
            return self._handle_implicit_confirmation(system_act)
        elif system_act.type is AgentActionType.CLOSE:
            return UserActions.close.value
        else:
            raise ValueError("System-act of type {} is not supported"
                             .format(system_act.type.value))

    def _mark_all_slots_as_provided(self):
        """Sets the status of all slots as "PROVIDED".

        Returns:
            None
        """
        for id_ in xrange(NUM_SLOTS):
            self.state.slots[id_] = UserStateStatus.PROVIDED

    def _handle_greeting(self):
        # Update system_act in the user-state.
        self.state.system_act = AgentActionType.GREET

        # If the system greets, the user stays silent with probability p,
        # and provides all slots with probability (1-p).
        b = binomial(1, USER_SILENT_VS_ALL_SLOTS_PROBABILITY)
        if b == 1:
            action = UserActions.silent.value
        else:
            action = UserActions.all_slots.value
            self._mark_all_slots_as_provided()  # Mark all slots as PROVIDED
        return action

    def _handle_slot_request(self, system_act):
        # Update system_act in the user-state.
        self.state.system_act = AgentActionType.ASK_SLOT

        # Following a request for a slot from the system, the user either
        # provides the requested slot with probability p or provides info.
        # for all slots with probability (1-p).
        requested_slot_id = system_act.ask_id
        b = binomial(1, USER_ONE_SLOT_VS_ALL_SLOTS_PROBABILITY)
        if b == 1:
            action = UserActions.one_slot.value[requested_slot_id]
            # Mark the requested slot as "PROVIDED"
            self.state.slots[requested_slot_id] = UserStateStatus.PROVIDED
        else:
            action = UserActions.all_slots.value
            self._mark_all_slots_as_provided()  # Mark all slots as PROVIDED
        return action

    def _handle_explicit_confirmation(self, system_act):
        # Update system_act in the user-state.
        action = UserActions.confirm.value[system_act.confirm_id]

        # An explicit confirmation request from the system is always responded
        # to by the user with affirmative.
        self.state.system_act = AgentActionType.EXPLICIT_CONFIRM
        return action

    def _handle_implicit_confirmation(self, system_act):
        # Update system_act in the user-state.
        self.state.system_act = AgentActionType.CONFIRM_ASK

        # Following an implicit confirmation from the system, the user provides
        # the requested slot with probability p -- implicitly confirming the
        # slot being confirmed -- and negates the confirmation with probability
        # (1-p) without providing any new information.
        requested_slot_id = system_act.ask_id
        confirmation_slot_id = system_act.confirm_id
        b = binomial(1, USER_ONE_SLOT_VS_NEGATE_PROBABILITY)
        if b == 1:
            action = UserActions.one_slot.value[requested_slot_id]
            # Mark the requested slot as "PROVIDED"
            self.state.slots[requested_slot_id] = UserStateStatus.PROVIDED
            # Mark the slot whose confirmation is sought implicitly
            # as "CONFIRMED"
            self.state.slots[confirmation_slot_id] = UserStateStatus.CONFIRMED
        else:
            action = UserActions.negate.value[requested_slot_id]
        return action
