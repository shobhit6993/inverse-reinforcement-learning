import pickle
import numpy as np

from agent.agent import Agent
from dialog_session import DialogSession
from mdp.solver import QLearningSolver
from user.user import User
from user.user_features import UserFeatures
from user_simulation import UserSimulation
from utils.params import UserPolicyType, GAMMA, NUM_SESSIONS_FE, THRESHOLD
from utils.params import SIMULATIONS_DUMP_FILE


class IRL(object):
    """The Inverse Reinforcment Learning class for building a user simulation
    in dialog systems.

    Attributes:
        agent (Agent): The dialog agent class
        features (UserFeatures): Feature function for dialog users.
        real_user (:obj: User): An expert user with a hand-crafted dialog
            policy.
        simulated_users (list of :obj: UserSimulation): List of user
            simulations built during the IRL algorithm.
        user (User): The dialog user class.
    """

    def __init__(self):
        self.user = User
        self.agent = Agent
        self.real_user = self.user(UserPolicyType.handcrafted)
        self.simulated_users = []
        self.features = UserFeatures()

    def run_irl(self):
        """Executes Inverse Reinforcement Learning algorithm to learn a set of
        decent user simulations. One among these is the best.
        """

        # The algorithm and the terminology here is based on the "Simpler
        # Algorithm" in Section 3.1 of Abbeel and Ng 2004 paper titled:
        # "Apprenticeship Learning via Inverse Reinforcement Learning."

        # Calculate feature expectation for the expert user policy.
        mu_e = self._calc_feature_expectation(self.real_user, self.agent())

        # Start with a user simulation with random policy.
        random_user = self.user(UserPolicyType.random)

        # Calculate feature expectation for the random user policy.
        mu_curr = self._calc_feature_expectation(random_user, self.agent())

        mu_bar_curr = mu_curr
        w = mu_e - mu_bar_curr  # The weight vector.
        t = np.linalg.norm(mu_e - mu_bar_curr)  # Margin of separation.

        print w
        print t

        # The learned weights w define a reward function. This reward function
        # is somewhat close to the expert's reward function. Learn an optimal
        # policy for that reward function, resulting in a decent simulated
        # user.
        sim_user = self.user()
        q_learning = QLearningSolver(sim_user, self.agent(), w)
        q_learning.solve()

        print "\nQ-values"
        print q_learning.q
        print "\n Policy"
        print sim_user.policy.policy
        print "--------------------------------"

        # Calculate feature expectation of the new policy.
        mu_curr = self._calc_feature_expectation(sim_user, self.agent())

        # Save the simulated user.
        self._save_simulated_user(sim_user, w, q_learning.q,
                                  mu_e, mu_curr)

        mu_bar_prev = mu_bar_curr

        steps = 0
        while t >= THRESHOLD:
            print("Step-{}".format(steps))
            # Dump the list of user simulations every 10 step.
            if steps % 10 == 0:
                self._dump_simulations()

            numerator = np.dot((mu_curr - mu_bar_prev), (mu_e - mu_bar_prev))
            denominator = np.dot((mu_curr - mu_bar_prev),
                                 (mu_curr - mu_bar_prev))
            factor = mu_curr - mu_bar_prev

            mu_bar_curr = mu_bar_prev + (numerator / denominator) * factor
            w = mu_e - mu_bar_curr
            t = np.linalg.norm(mu_e - mu_bar_curr)

            print w
            print t

            # The learned weights w define a reward function. This reward
            # function is somewhat close to the expert's reward function.
            # Learn an optimal policy for that reward function, resulting
            # in a decent simulated user.
            sim_user = self.user()
            q_learning = QLearningSolver(sim_user, self.agent(), w)
            q_learning.solve()

            print "\nQ-values"
            print q_learning.q
            print "\n Policy"
            print sim_user.policy.policy
            print "--------------------------------"

            # Calculate feature expectation of the new policy.
            mu_curr = self._calc_feature_expectation(sim_user, self.agent())

            # Save the simulated user.
            self._save_simulated_user(random_user, w, q_learning.q,
                                      mu_e, mu_curr)

            mu_bar_prev = mu_bar_curr
            steps += 1

        # Dump the final list of user simulations.
        if steps % 10 == 0:
            self._dump_simulations()

    def _calc_feature_expectation(self, user, agent,
                                  num_sessions=NUM_SESSIONS_FE):
        """Calculates the feature expectation of a user policy against the
        handcoded agent by executing a series of dialog sessions and tracking
        the state-action pairs associated with the user.

        Args:
            user (:obj: User): The user whose policy's feature expectation
                needs to be calculated.
            agent (:obj: Agent): The agent against whom the `user` the dialog
                sessions will be run.
            num_sessions (int, optional): Number of dialog sessions to be run
                for the purpose of feature expectation calculation.

        Returns:
            numpy.array: Feature expectation of the user's policy.
        """
        feature_expectation = np.zeros(self.features.dimensions)
        for _ in xrange(num_sessions):
            session = DialogSession(user, agent)
            session.start()

            for t, (state, action) in enumerate(session.user_log):
                feature_vector = self.features.get_vector(state, action)
                feature_expectation += ((GAMMA**t) * feature_vector)

        feature_expectation /= num_sessions
        return feature_expectation

    def _save_simulated_user(self, user, weights, q, expert_fe, simulated_fe):
        """Saves the simulated user built during an iteration of IRL algorithm.

        Args:
            user (:obj: User): The learnt user simulation.
            weights (1D numpy.ndarray): The weight vectors characterizing the
                `Reward` function that gave rise to this user-simulation.
            q (dict): Q-value function of the user policy.
            expert_fe (1d numpy.ndarray): Expert user's feature expectations.
            simulated_fe (1d numpy.ndarray): Simulated user's feature
                expectations.
        """
        distance_to_expert = np.linalg.norm(expert_fe - simulated_fe)
        simulated_user = UserSimulation(user.policy, q, weights,
                                           distance_to_expert)
        self.simulated_users.append(simulated_user)

    def _dump_simulations(self):
        """Dumps the list of user simulations, i.e., the
        `IRL.simulated_users` attribute.
        """
        with open(SIMULATIONS_DUMP_FILE, "w") as fout:
            pickle.dump(self.simulated_users, fout)
