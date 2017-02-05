import numpy as np


class Preference(object):
    """Prefernce class for MDP.

    Prefernce function is assumed to be a linear combination of the feature
    vector for the given state-action pair, parameterized by the weights
    governing the linear combination.

    Attributes:
        features (UserFeature): Feature function for the RL agent (here user).
        weights (numpy.array): Weights parameterizing the preference function.
    """

    def __init__(self, features, theta=None):
        self.features = features
        self.theta = self._initialize_theta(theta)

    def get_preference(self, state, action):
        """Returns the preference for taking the action in the given state.

        Args:
            state (AgentActionType): State of the user characterized by
                AgentActionType -- the last action of the dialog agent.
            action (UserActionType): Type of the action taken by the user.

        Returns:
            float: Preference value for the state-action pair
        """

        feature_vector = self.features.get_vector(state, action)
        return np.dot(self.theta, feature_vector)

    def _initialize_theta(self, theta):
        n = self.features.dimensions
        if type(theta) is np.ndarray and len(theta) == n:
            return theta
        else:
            return 0.5 * np.ones(n)
