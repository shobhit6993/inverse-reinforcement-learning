class UserSimulation(object):
    """User simulation learnt using IRL.

    Attributes:
        distance_to_expert (float): Difference between the feature expectations
            of the expert user and the simulated user.
        policy (:obj: UserPolicy): Policy defining the user simulation
        q (dict): Q-values of the policy.
        weights (1D numpy.ndarray): The weight vectors characterizing the
            `Reward` function that gave rise to this user-simulation.
    """

    def __init__(self, policy, q, weights, distance):
        """Constructor for UserSimulation class

        Args:
            policy (:obj: UserPolicy): Policy defining the user simulation
            q (dict): Q-values of the policy.
            weights (1D numpy.ndarray): The weight vectors characterizing the
                `Reward` function that gave rise to this user-simulation.
            distance (float): Eucledian distance between the feature
                expectations of the expert user and the simulated user.
        """
        self.policy = policy
        self.q = q
        self.weights = weights
        self.distance_to_expert = distance
