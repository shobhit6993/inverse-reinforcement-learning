"""Dialog user."""

from numpy.random import randint

from user_action import UserActions
from user_features import UserFeatures
from user_policy import UserPolicy
from user_state import UserState
from utils.params import AgentActionType, NUM_SLOTS
from utils.params import UserActionType, UserStateStatus


class User(object):
    """Class for the user in a dialog system. It keeps track of the user's
    state and picks actions in response to agent's actions using some policy.

    Attributes:
        features (UserFeatures): Feature class for user's state-action space.
        policy (UserPolicy): Policy to be followed by the user.
        state (UserState): User's current state.
    """

    def __init__(self, policy_type):
        """Class constructor

        Args:
            policy_type (UserPolicyType): Type of user policy.
        """
        self.state = UserState()
        self.policy = UserPolicy(policy_type)
        self.features = UserFeatures()

    def take_turn(self, system_act):
        """Executes a user turn based on the agent's most recent action.

        Args:
            system_act (AgentAction): Dialog agent's most recent action.

        Returns:
            UserAction: User's next action.
        """
        next_action = self.update_state_and_get_next_action(system_act)
        print("User -- [State] " + str(self.state))
        print("User -- (Action) " + str(next_action))
        # print("U:" + next_action.type.value)
        return next_action

    def update_state_and_get_next_action(self, system_act):
        """Updates the user-state and returns the next action to be taken.
        The state-update and next action are based on the user-`policy` and
        the agent's most recent action

        Args:
            system_act (AgentAction): Dialog agent's most recent action.

        Returns:
            UserAction: User's next action.
        """
        # Partially update state by updating the `system_act`.
        self.state.system_act = system_act

        # From the policy, sample the type of action, a UserActionType, to be
        # taken.
        action_type = self.policy.get_action(self.state)

        # Build the full UserAction based on the sampled action type.
        action = self._build_action(action_type)

        # Update state to reflect the action that is about to be taken.
        self._update_state(action)
        return action

    def _build_action(self, action_type):
        """Builds a full action based on the sampled action type.
        The action object -- UserAction -- is based on the UserActionType
        returned by the policy, and contains other details such as slot ids.

        Args:
            action_type (UserActionType): The type of user-action to be taken.

        Returns:
            UserAction: The action to be taken.
        """
        requested_slot_id = self.state.system_act.ask_id
        confirm_slot_id = self.state.system_act.confirm_id
        random_slot_id = randint(NUM_SLOTS)

        if action_type is UserActionType.SILENT:
            return UserActions.silent.value

        elif action_type is UserActionType.ALL_SLOTS:
            return UserActions.all_slots.value

        elif action_type is UserActionType.ONE_SLOT:
            # If agent asked for a slot, return UserAction corresponding to
            # that slot. Otherwise, return UserAction corresponding to a
            # a random slot.
            if requested_slot_id is not None:
                return UserActions.one_slot.value[requested_slot_id]
            else:
                return UserActions.one_slot.value[random_slot_id]

        elif action_type is UserActionType.CONFIRM:
            return UserActions.confirm.value[confirm_slot_id]
        elif action_type is UserActionType.NEGATE:
            return UserActions.negate.value[confirm_slot_id]
        elif action_type is UserActionType.CLOSE:
            return UserActions.close.value

    def _update_state(self, action):
        """Updates the user-state based on the action about to be taken.

        Args:
            action (UserAction): Action the user is about to take.
        """
        slot_id = action.slot_id

        if action.type is UserActionType.SILENT:
            pass

        elif action.type is UserActionType.ALL_SLOTS:
            # Mark all slots as PROVIDED
            self._mark_all_slots_as_provided()

        elif action.type is UserActionType.ONE_SLOT:
            # Mark the requested slot as "PROVIDED"
            self.state.slots[slot_id] = UserStateStatus.PROVIDED

            # If the agent asked for an implicit confirmation, and the user
            # responds with information for the requested slot, then in
            # addition to marking the requested slot as provided, mark the slot
            # being implicitly confirmed as "CONFIRMED".
            if self.state.system_act.type is AgentActionType.CONFIRM_ASK:
                confirm_slot_id = self.state.system_act.confirm_id
                self.state.slots[confirm_slot_id] = UserStateStatus.CONFIRMED

        elif action.type is UserActionType.CONFIRM:
            # Mark the slot being confirmed as "CONFIRMED"
            self.state.slots[slot_id] = UserStateStatus.CONFIRMED

        elif action.type is UserActionType.NEGATE:
            # Mark the slot being negated as "EMPTY"
            self.state.slots[slot_id] = UserStateStatus.EMPTY

        elif action.type is UserActionType.CLOSE:
            pass

    def _mark_all_slots_as_provided(self):
        """Sets the status of all slots as "PROVIDED".
        """
        for id_ in xrange(NUM_SLOTS):
            self.state.slots[id_] = UserStateStatus.PROVIDED
