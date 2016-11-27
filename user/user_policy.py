"""Dialog policy for users."""

import numpy as np

from user_features import UserFeatures
from utils.params import AgentActionType, UserActionType, UserPolicyType
from utils import utils


class UserPolicy(object):
    """Policy class for user.

    A policy is defined as a mapping from user-state to the type of action
    that the user should perform in that state. While `UserState` -- which
    encodes the user's state -- consists of other attributes, only the
    `UserState.agent_act` -- an `AgentActionType` enum -- is consequential
    to the user's policy. Hence, policy is a mapping from `AgentActionType` to
    `UserActionType`.

    Attributes:
        action_index_map (dict): Mapping from a `UserActionType` to it's index
            in the `actions` attribute.
        actions (dict): List of possible action types, i.e., UserActionType
            enum members.
        policy (dict): The defined policy. It is a dictionary indexed by state.
            Since `UserState.agent_act` -- an `AgentActionType` enum -- is the
            only consequential part of user's state, the dictionary is indexed
            by it. The value of each entry is a list of probability values.
            Each entry in the list is the probability, in that state, of
            choosing the corresponding action in `actions`.
    """

    def __init__(self, policy_type=None):
        """Class constructor

        Args:
            policy_type (UserPolicyType, optional): Type of user policy; None
                default. None means a valid policy would not be build during
                creation.
        """
        self.actions = [user_action for user_action in UserActionType]
        self.action_index_map = {action: i for i, action in
                                 enumerate(self.actions)}
        self.policy = {action_type: np.zeros(len(self.actions))
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
        state = user_state.agent_act.type
        probabilities = self.policy[state]
        sampled_action = np.random.choice(self.actions, 1, p=probabilities)[0]
        return sampled_action   # a UserActionType

    def build_policy_from_q_values(self, q_function, epsilon):
        """Defines an epsilon-greedy policy derived from the Q-values.

        Args:
            q_function (dict): The Q-value function. It must have the same
                structure as the `policy` attribute, except that the values in
                the lists represent Q-values.
            epsilon (float): Degree of randomness required in the policy.
        """
        assert len(q_function) == len(self.policy)
        for state in self.policy:
            assert state in q_function
            assert type(q_function[state]) is np.ndarray
            assert q_function[state].shape == self.policy[state].shape

        for state in self.policy:
            q_values = q_function[state]
            n = len(q_values)

            probabilities = self.policy[state]
            # Assign epsion probability mass to each action.
            for i in xrange(n):
                probabilities[i] = epsilon

            # Assign the remaining probability mass to the action with
            # highest q value.
            best_action_index = utils.get_index_of_max_element(q_values)
            probabilities[best_action_index] += 1.0 - np.sum(probabilities)

            # Make sure that the probabilities sum to 1.
            # If they don't -- due to precision issues -- keep adding the
            # difference to some element until it they do.
            sum_of_probabilities = np.sum(probabilities)
            while sum_of_probabilities != 1.0:
                diff = 1.0 - sum_of_probabilities
                random_index = np.random.randint(n)
                probabilities[random_index] += diff
                sum_of_probabilities = np.sum(probabilities)

        self._check_policy_correctness()

    def reset(self):
        """Resets the policy to an invalid, all-zero-probabilities policy."""
        for state in self.policy:
            for i in xrange(len(self.actions)):
                self.policy[state][i] = 0.

    def _build_policy(self, policy_type):
        """Builds a user policy.

        Args:
            policy_type (UserPolicyType or None): Type of user policy.
        """
        if policy_type is UserPolicyType.handcrafted:
            self._build_handcrafted_policy()
        elif policy_type is UserPolicyType.random:
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
            assert np.sum(probabilities) == 1.0, ("Probabilities don't sum to "
                                                  "1 for {}".format(state))
