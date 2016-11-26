"""Dialog policy for users."""

import numpy as np

from user_features import UserFeatures
from utils.params import AgentActionType, UserActionType, UserPolicyType


class UserPolicy(object):
    """Policy class for user.

    A policy is defined as a mapping from user-state to the type of action
    that the user should perform in that state. While `UserState` -- which
    encodes the user's state -- consists of other attributes, only the
    `UserState.system_act` -- an `AgentActionType` enum -- is consequential
    to the user's policy. Hence, policy is a mapping from `AgentActionType` to
    `UserActionType`.

    Attributes:
        action_index_map (dict): Mapping from a `UserActionType` to it's index
            in the `actions` attribute.
        actions (dict): List of possible action types, i.e., UserActionType
            enum members.
        policy (dict): The defined policy. It is a dictionary indexed by state.
            Since `UserState.system_act` -- an `AgentActionType` enum -- is the
            only consequential part of user's state, the dictionary is indexed
            by it. The value of each entry is a list of probability values.
            Each entry in the list is the probability, in that state, of
            choosing the corresponding action in `actions`.
    """

    def __init__(self, policy_type):
        """Class constructor

        Args:
            policy_type (UserPolicyType): Type of user policy
        """
        self.actions = [user_action for user_action in UserActionType]
        self.action_index_map = {action: i for i, action in
                                 enumerate(self.actions)}
        self.policy = {action_type: [0.] * len(self.actions)
                       for action_type in AgentActionType}

        self._build_policy(policy_type)

    def get_action(self, user_state):
        """Samples the type of action to be taken from the policy given
        current state.

        Args:
            user_state (UserState): User state.

        Returns:
            UserActionType: Type of the action to be taken.
        """
        state = user_state.system_act.type
        probabilities = self.policy[state]
        sampled_action = np.random.choice(self.actions, 1, p=probabilities)[0]
        return sampled_action   # a UserActionType

    def _build_policy(self, policy_type):
        """Builds a user policy.

        Args:
            policy_type (UserPolicyType): Type of user policy.
        """
        if policy_type == UserPolicyType.handcrafted:
            self._build_handcrafted_policy()
        elif policy_type == UserPolicyType.random:
            self._build_random_policy()

    def _build_handcrafted_policy(self):
        """Defines a hand-crafter policy for the user.
        """
        silent = self.action_index_map[UserActionType.SILENT]
        all_slots = self.action_index_map[UserActionType.ALL_SLOTS]
        one_slot = self.action_index_map[UserActionType.ONE_SLOT]
        confirm = self.action_index_map[UserActionType.CONFIRM]
        negate = self.action_index_map[UserActionType.NEGATE]
        close = self.action_index_map[UserActionType.CLOSE]

        self.policy[AgentActionType.GREET][silent] = 0.7
        self.policy[AgentActionType.GREET][all_slots] = 0.3

        self.policy[AgentActionType.ASK_SLOT][one_slot] = 0.95
        self.policy[AgentActionType.ASK_SLOT][all_slots] = 0.05

        self.policy[AgentActionType.EXPLICIT_CONFIRM][confirm] = 1.0

        self.policy[AgentActionType.CONFIRM_ASK][one_slot] = 0.9
        self.policy[AgentActionType.CONFIRM_ASK][negate] = 0.1

        self.policy[AgentActionType.CLOSE][close] = 1.0

        self._check_policy_correctness()

    def _build_random_policy(self):
        """Defines a random policy.
        """
        alpha = 5 * np.ones(len(self.actions))  # Parameter for Dirichlet
        for state in self.policy:
            probabilities = np.random.dirichlet(alpha).tolist()
            self.policy[state] = probabilities

    def _check_policy_correctness(self):
        """Validates the defined policy by checking for valid probablity
        values.
        """
        for state, probabilities in self.policy.iteritems():
            assert sum(probabilities) == 1.0, ("Probabilities don't sum"
                                               "to 1 for", state)
