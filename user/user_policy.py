"""Dialog policy for the users."""

from numpy.random import choice

from utils.params import AgentActionType
from utils.params import UserActionType


class UserPolicy(object):
    def __init__(self):
        # List of possible action types -- UserActionType enum members.
        self.actions = [user_action for user_action in UserActionType]

        # Mapping from action to it's index in the `actions` list.
        self.action_index_map = {action: i for i, action in
                                 enumerate(self.actions)}

        # `states` is a dictionary indexed by user-state. Since
        # `UserState.system_act` -- an `AgentActionType` enum -- is the
        # only consequential part of a user's state, the dictionary is indexed
        # by it. The value of each entry is a list of probability values.
        # Each entry in the list is the probability, in that state, of choosing
        # the corresponding action in `actions`.
        self.states = {action_type: [0.] * len(self.actions)
                       for action_type in AgentActionType}

        self._build_policy()

    def get_action(self, user_state):
        #user_state = UserState
        state = user_state.system_act.type
        probabilities = self.states[state]
        sampled_action = choice(self.actions, 1, p=probabilities)[0]
        return sampled_action   # a UserActionType

    def _build_policy(self):
        silent = self.action_index_map[UserActionType.SILENT]
        all_slots = self.action_index_map[UserActionType.ALL_SLOTS]
        one_slot = self.action_index_map[UserActionType.ONE_SLOT]
        confirm = self.action_index_map[UserActionType.CONFIRM]
        negate = self.action_index_map[UserActionType.NEGATE]
        close = self.action_index_map[UserActionType.CLOSE]

        self.states[AgentActionType.GREET][silent] = 0.7
        self.states[AgentActionType.GREET][all_slots] = 0.3

        self.states[AgentActionType.ASK_SLOT][one_slot] = 0.95
        self.states[AgentActionType.ASK_SLOT][all_slots] = 0.05

        self.states[AgentActionType.EXPLICIT_CONFIRM][confirm] = 1.0

        self.states[AgentActionType.CONFIRM_ASK][one_slot] = 0.9
        self.states[AgentActionType.CONFIRM_ASK][negate] = 0.1

        self.states[AgentActionType.CLOSE][close] = 1.0

        self._check_policy_correctness()

    def _check_policy_correctness(self):
        for state, probabilities in self.states.iteritems():
            assert sum(probabilities) == 1.0, ("Probabilities don't sum"
                                               "to 1 for", state)
