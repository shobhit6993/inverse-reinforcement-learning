import numpy as np

from utils.params import AgentActionType, UserActionType


class UserFeatures(object):
    def __init__(self):
        self._function = {}

        self._define_function()

    def get_vector(self, state, action):
        """Returns the feature vector for a state-action pair.

        Args:
            state (AgentActionType): User state.
            action (UserActionType): User action.

        Returns:
            numpy.array: Feature vector for the given state-action pair.
        """
        try:
            return self._function[(state, action)]
        except KeyError as e:
            print("Exception: {}. Invalid state: '{}' and action: '{}' pair"
                  .format(e, state, action))
            raise

    def _define_function(self):
        """Builds the feature function.
        """
        # First calculate the size of input space.
        n = 0
        # User-state is effectively characterized by AgentActionType
        for state in AgentActionType:
            for action in UserActionType:
                n += 1

        # Now construct the feature function -- mapping from state-action pairs
        # to an n dim vector whose elements are in the range [0, 1]
        i = 0
        for state in AgentActionType:
            for action in UserActionType:
                vec = np.zeros(n)
                vec[i] = 1.
                self._function[(state, action)] = vec
                i += 1
