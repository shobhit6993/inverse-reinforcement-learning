class UserSimulationIRL(object):
    """User simulation learnt using IRL.

    Attributes:
        distance_to_expert (float): Difference between the feature expectations
            of the expert user and the simulated user.
        policy (:obj: UserPolicy): Policy defining the user simulation
        weights (1D numpy.ndarray): The weight vectors characterizing the
            `Reward` function that gave rise to this user-simulation.
    """

    def __init__(self, policy, weights, distance):
        """Constructor for UserSimulationIRL class

        Args:
            policy (:obj: UserPolicy): Policy defining the user simulation
            weights (1D numpy.ndarray): The weight vectors characterizing the
                `Reward` function that gave rise to this user-simulation.
            distance (float): Difference between the feature expectations
                of the expert user and the simulated user.
        """
        self.policy = policy
        self.weights = weights
        self.distance_to_expert = distance
