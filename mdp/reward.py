import numpy as np


class Reward(object):
    """Reward class for MDP.

    Reward function is assumed to be a linear combination of the feature
    vector for the given state-action pair, parameterized by the weights
    governing the linear combination.

    Attributes:
        features (UserFeature): Feature function for the RL agent (here user).
        weights (numpy.array): Weights parameterizing the reward function.
    """

    def __init__(self, features, weights):
        self.features = features
        self.weights = weights

    def get_reward(self, state, action):
        """Returns the reward for taking the action in the given state.

        Args:
            state (AgentActionType): State of the user characterized by
                AgentActionType -- the last action of the dialog agent.
            action (UserActionType): Type of the action taken by the user.

        Returns:
            TYPE: Description
        """
        feature_vector = self.features.get_vector(state, action)
        return np.dot(self.weights, feature_vector)
