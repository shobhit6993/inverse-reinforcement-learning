from user.user import User


class UserSimulation(User):
    """User simulation learnt using IRL.

    Attributes:
        distance_to_expert (float): Difference between the feature expectations
            of the expert user and the simulated user.
        policy (:obj: UserPolicy): Policy defining the user simulation
        q (dict): Q-values of the policy.
        weights (1D numpy.ndarray): The weight vectors characterizing the
            `Reward` function that gave rise to this user-simulation.
    """

    def __init__(self, policy=None, q=None, weights=None,
                 distance_to_expert=None):
        super(UserSimulation, self).__init__(policy=policy)
        # self.policy = policy
        self.q = q
        self.weights = weights
        self.distance_to_expert = distance_to_expert

    def __str__(self):
        return ("Distance: {} \n Policy: {}"
                .format(str(self.distance_to_expert), str(self.policy)))
