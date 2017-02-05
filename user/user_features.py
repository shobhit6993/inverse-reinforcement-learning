import numpy as np

from utils.params import AgentActionType, UserActionType


def deco(cls):
    cls._define_function()
    return cls


@deco
class UserFeatures(object):
    dimensions = 0
    _function = {}

    @classmethod
    def get_vector(cls, state, action):
        """Returns the feature vector for a state-action pair.

        Args:
            state (AgentActionType): User state.
            action (UserActionType): User action.

        Returns:
            numpy.array: Feature vector for the given state-action pair.
        """
        try:
            return cls._function[(state, action)]
        except KeyError as e:
            print("Exception: {}. Invalid state: '{}' and action: '{}' pair"
                  .format(e, state, action))
            raise

    @classmethod
    def _define_function(cls):
        """Builds the feature function.
        """
        # First calculate the size of input space.
        cls.dimensions = 0
        # User-state is effectively characterized by AgentActionType
        for state in AgentActionType:
            for action in UserActionType:
                cls.dimensions += 1

        # Now construct the feature function -- mapping from state-action pairs
        # to an n dim vector whose elements are in the range [0, 1]
        i = 0
        for state in AgentActionType:
            for action in UserActionType:
                vec = np.zeros(cls.dimensions)
                vec[i] = 1.
                cls._function[(state, action)] = vec
                i += 1
