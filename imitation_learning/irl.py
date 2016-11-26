import numpy as np

from agent.agent import Agent
from dialog_session import DialogSession
from user.user import User
from user.user_features import UserFeatures
from utils.params import UserPolicyType, GAMMA, NUM_SESSIONS


class IRL(object):
    """The Inverse Reinforcment Learning class for building a user simulation
    in dialog systems.

    Attributes:
        agent (Agent): The dialog agent.
        features (UserFeatures): Feature function for dialog users.
        real_user (User): An expert user with a hand-crafted dialog policy.
        sim_user (Usr): A simulated user that is initilized with a random
            policy. The aim is to have this user simulate the unknown policy
            of the expert user.
    """

    def __init__(self):
        self.sim_user = User(UserPolicyType.random)
        self.real_user = User(UserPolicyType.handcrafted)
        self.agent = Agent()
        self.features = UserFeatures()

    def calc_feature_expectation(self, user, num_sessions=NUM_SESSIONS):
        """Calculates the feature expectation of a user policy against the
        handcoded agent by executing a series of dialog sessions and tracking
        the state-action pairs associated with the user.

        Args:
            user (User): The user whose policy's feature expectation needs to
                be calculated.
            num_sessions (int, optional): Number of dialog sessions to be run
                for the purpose of feature expectation calculation.

        Returns:
            numpy.array: Feature expectation of the user's policy.
        """
        feature_expectation = np.zeros(self.features.dimensions)
        for _ in xrange(num_sessions):
            session = DialogSession(user, self.agent)
            session.start()

            for t, (state, action) in enumerate(session.user_log):
                feature_vector = self.features.get_vector(state, action)
                feature_expectation += ((GAMMA**t) * feature_vector)

        feature_expectation /= num_sessions
        return feature_expectation
