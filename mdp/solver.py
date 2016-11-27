from abc import abstractmethod
from copy import deepcopy
import numpy as np

from reward import Reward
from imitation_learning.dialog_session import DialogSession
from utils.params import DECAY_RATE, Q_LEARNING_EPISODES, EPSILON
from utils.params import AgentActionType


class MDPSolver(object):
    """Metaclass for MDP solver.

    Given the environment (dialog agent), RL agent (user), and reward function,
    it solves the MDP problem to obtain a near-optimal policy for the RL agent.

    Attributes:
        agent (:obj: Agent): The dialog agent, which acts as the environment
            for the MDP solver.
        reward (:obj: Reward): The Reward function.
        user (:obj: User): The dialog user, which acts as the RL agent for
            which a near-optimal policy is desired under the given reward
            function.
        weights (1D numpy.ndarray): Weight vector parameterizing the reward
            function.
    """

    def __init__(self, user, agent, weights):
        self.user = user
        self.agent = agent
        self.weights = weights
        self.reward = Reward(self.user.features, self.weights)

    @abstractmethod
    def solve(self):
        pass


class QLearningSolver(MDPSolver):
    """Q-Learning class for solving an MDP.

    Attributes:
        alpha (float): Learning rate
        gamma (float): Discount factor
        q (dict): Q-value function. The structure of this function should
            be exactly same as that of the `UserPolicy.poliy` attribute.
        terminal_state (AgentActionType): Terminal state for RL agent (user).
    """

    def __init__(self, user, agent, weights):
        super(QLearningSolver, self).__init__(user, agent, weights)

        self.alpha = 0.1
        self.gamma = 0.95
        self.terminal_state = AgentActionType.CLOSE
        self.q = {}

        self._initialize_q_values()

    def solve(self):
        """Executes Q-learning to learn a near-optimal policy for the MDP.
        """
        for _ in xrange(Q_LEARNING_EPISODES):
            # Reset the agent and the user.
            self.user.reset()
            self.agent.reset()

            # Create a new dialog session.
            session = DialogSession(self.user, self.agent)

            # Update user's policy based on the updated Q-values.
            self.user.policy.build_policy_from_q_values(self.q, EPSILON)

            # Start the dialog session by having the dialog agent make the
            # first move.
            agent_first_action = session.ask_agent_to_start()

            # State of the  user is characterized by the type of action taken
            # taken by the agent -- an AgentActionType.
            curr_state = agent_first_action.type

            while curr_state is not self.terminal_state:
                action, next_state = session.execute_one_step()
                reward = self.reward.get_reward(curr_state, action)
                self._update_q_value(curr_state, action, next_state, reward)
                curr_state = next_state

            self.alpha *= DECAY_RATE

    def _initialize_q_values(self):
        """Initializes Q-values.
        """
        # Replicate the structure of user's policy from `UserPolicy.policy`
        self.q = deepcopy(self.user.policy.policy)
        num_actions = len(self.user.policy.actions)

        # Initialize all q-values with the same value.
        for state in self.q:
            self.q[state] = 0.5 * np.ones(num_actions)

        self.q[self.terminal_state] = np.zeros(num_actions)

    def _update_q_value(self, state, action, next_state, reward):
        """Performs a TD(0) update to the Q-value function.

        Args:
            state (AgentActionType): User's current state.
            action (UserActionType): Action taken by the user.
            next_state (AgentActionType): User's next state
            reward (float): Reward received by the user during this transition.
        """
        max_q_value = max(self.q[next_state])
        action_ix = self.user.policy.action_index_map[action]
        td_error = reward + self.gamma * max_q_value - self.q[state][action_ix]
        self.q[state][action_ix] += self.alpha * td_error
