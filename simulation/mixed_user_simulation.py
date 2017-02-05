import pickle
import numpy as np
import cvxopt as cvx

from agent.agent import Agent
from imitation_learning.dialog_session import DialogSession
from imitation_learning.irl import IRL
from user_simulation import UserSimulation
from user.user_features import UserFeatures
from utils.params import AgentActionType, UserActionType, TAU
from utils import utils

def run_single_session(user):
    """Executes a single dialog session.
    """
    agent = Agent()
    user.reset(reset_policy=False)
    session = DialogSession(user, agent)
    session.start()
    return session.user_log

class QpMixedUserSimulation(UserSimulation):
    def __init__(self, filepath, real_user):
        super(QpMixedUserSimulation, self).__init__()
        self.users = self._load_user_simulations(filepath)
        self.real_user = real_user
        self.mixture_weights = None

    def solve_qp(self):
        num_users = len(self.users)

        # Calculate feature expectations of all simulated users.
        fe = []
        for user in self.users:
            fe.append(IRL.calc_feature_expectation(user, Agent()))

        # Calculate feature expectation of expert user.
        fe_expert = IRL.calc_feature_expectation(self.real_user, Agent())

        # Cacluate matrix P in QP formulation of cvxopt
        P = np.zeros((num_users, num_users))
        for i in xrange(num_users):
            for j in xrange(i, num_users):
                product = np.dot(fe[i], fe[j])
                P[i][j] = product
                P[j][i] = product
        P = cvx.matrix(P)
        # print P

        q = np.zeros(num_users)
        for i in xrange(num_users):
            q[i] = -2 * np.dot(fe_expert, fe[i])
        q = cvx.matrix(q)
        # print q

        G = np.eye(num_users) * (-1)
        G = cvx.matrix(G)
        # print G

        h = np.zeros(num_users)
        h = cvx.matrix(h, (num_users, 1))
        # print h

        A = np.ones(num_users)
        A = cvx.matrix(A, (1, num_users))
        # print A

        b = cvx.matrix([1.], (1, 1))
        # print b

        sol = cvx.solvers.qp(P, q, G, h, A, b)
        print sol

        self.mixture_weights = np.array(sol['x']).reshape(num_users)
        utils.normalize_probabilities(self.mixture_weights)

    def collect_statistics(self, num_sessions):
        """Runs multiple dialog sessions between the user and the agent to collect
        statistics about user's actions.

        Args:
            user (:obj: User): The dialog user.
            agent (:obj: Agent): The dialog agent.
            dialog_session (DialogSession): The dialog session class
            num_sessions (int): Number of dialog sessions to execute.
        """
        user_actions = [user_action for user_action in UserActionType]
        user_action_map = {action: i for i, action in
                           enumerate(user_actions)}
        user_action_stats = {action_type: np.zeros(len(user_actions))
                             for action_type in AgentActionType}

        agent_actions = [agent_action for agent_action in AgentActionType]
        agent_action_map = {action: i for i, action in
                            enumerate(agent_actions)}
        agent_action_counts = np.zeros(len(agent_actions))

        agent = Agent()
        # Run multiple dialog sessions to gather user's action statistics.
        for _ in xrange(num_sessions):
            # Reset the agent and the user.
            agent.reset()
            user = self._pick_user_stochastically()
            user.reset(reset_policy=False)  # Only reset state, not policy.
            # Create a new dialog session.
            session = DialogSession(user, agent)
            # Start the dialog session by having the dialog agent make the
            # first move.
            agent_action = session.ask_agent_to_start()
            user_action = None
            while not (agent_action is AgentActionType.CLOSE and
                       user_action is UserActionType.CLOSE):
                user_action, next_agent_action = session.execute_one_step()

                # Update action statistics
                user_action_index = user_action_map[user_action]
                user_action_stats[agent_action][user_action_index] += 1
                agent_action_index = agent_action_map[agent_action]
                agent_action_counts[agent_action_index] += 1

                agent_action = next_agent_action

        print user_action_stats
        print agent_action_counts

    def temp(self, num_sessions):
        freq = {}
        for action in UserActionType:
            freq[action] = 0

        for _ in xrange(num_sessions):
            user = self._pick_user_stochastically()
            user_log = run_single_session(user)
            for _, action in user_log:
                freq[action] += 1

        for action in freq.keys():
            freq[action] /= (1. * num_sessions)

        return freq

    def _pick_user_stochastically(self):
        return np.random.choice(self.users, 1, p=self.mixture_weights)[0]

    def _load_user_simulations(self, filepath):
        with open(filepath, "r") as fin:
            user_simulations = pickle.load(fin)
            return user_simulations


class GibbsMixedUserSimulation(UserSimulation):
    def __init__(self, filepath):
        super(GibbsMixedUserSimulation, self).__init__()
        self.users = self._build_users_dictionary(filepath)
        self.probabilities = self._build_probability_dictionary()
        print self.users
        print self.probabilities

    def collect_statistics(self, num_sessions):
        """Runs multiple dialog sessions between the user and the agent to collect
        statistics about user's actions.

        Args:
            user (:obj: User): The dialog user.
            agent (:obj: Agent): The dialog agent.
            dialog_session (DialogSession): The dialog session class
            num_sessions (int): Number of dialog sessions to execute.
        """
        user_actions = [user_action for user_action in UserActionType]
        user_action_map = {action: i for i, action in
                           enumerate(user_actions)}
        user_action_stats = {action_type: np.zeros(len(user_actions))
                             for action_type in AgentActionType}

        agent_actions = [agent_action for agent_action in AgentActionType]
        agent_action_map = {action: i for i, action in
                            enumerate(agent_actions)}
        agent_action_counts = np.zeros(len(agent_actions))

        agent = Agent()
        # Run multiple dialog sessions to gather user's action statistics.
        for _ in xrange(num_sessions):
            # Reset the agent and the user.
            agent.reset()
            user = self._pick_user_stochastically()
            user.reset(reset_policy=False)  # Only reset state, not policy.
            # Create a new dialog session.
            session = DialogSession(user, agent)
            # Start the dialog session by having the dialog agent make the
            # first move.
            agent_action = session.ask_agent_to_start()
            user_action = None
            while not (agent_action is AgentActionType.CLOSE and
                       user_action is UserActionType.CLOSE):
                user_action, next_agent_action = session.execute_one_step()

                # Update action statistics
                user_action_index = user_action_map[user_action]
                user_action_stats[agent_action][user_action_index] += 1
                agent_action_index = agent_action_map[agent_action]
                agent_action_counts[agent_action_index] += 1

                agent_action = next_agent_action

        print user_action_stats
        print agent_action_counts

    def temp(self, num_sessions):
        freq = {}
        for action in UserActionType:
            freq[action] = 0

        for _ in xrange(num_sessions):
            user = self._pick_user_stochastically()
            user_log = run_single_session(user)
            for _, action in user_log:
                freq[action] += 1

        for action in freq.keys():
            freq[action] /= (1. * num_sessions)

        return freq

    def _pick_user_stochastically(self):
        users = []
        probabilities = []
        for k in self.users:
            users.append(self.users[k])
            probabilities.append(self.probabilities[k])
        utils.normalize_probabilities(probabilities)
        user = np.random.choice(users, 1, p=probabilities)[0]
        return user

    def _build_users_dictionary(self, filepath):
        user_simulations = self._load_user_simulations(filepath)
        distances = [(sim.distance_to_expert, i)
                     for i, sim in enumerate(user_simulations)]
        distances.sort()

        users = {}
        for (distance, i) in distances:
            dist_signature = "{:.3f}".format(distance)
            users[dist_signature] = user_simulations[i]
        return users

    def _build_probability_dictionary(self):
        s = sum([np.exp(-float(dist_string) / TAU)
                 for dist_string in self.users.keys()])
        probabilities = {}
        for dist_string in self.users.keys():
            probabilities[
                dist_string] = np.exp(-float(dist_string) / TAU) / s
        return probabilities

    def _load_user_simulations(self, filepath):
        with open(filepath, "r") as fin:
            user_simulations = pickle.load(fin)
            return user_simulations
