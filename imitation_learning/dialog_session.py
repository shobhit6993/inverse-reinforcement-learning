"""Dialog session."""

from agent.agent import Agent
from user.user import User
from user.user_action import UserAction
from utils.params import AgentActionType
from utils.params import UserActionType, UserPolicyType


class DialogSession:
    """Class for a single dialog session.

    Attributes:
        agent (:obj: Agent): The dialog agent.
        prev_agent_act (AgentAction): The previous action taken by the agent.
        user (:obj: User): The user participating in the dialog.
        user_log (list of tuples): Log of (state, action) pairs that the user
            underwent in this dialog session in the form of a list of tuples of
            form (AgentActionType, UserActionType).
    """

    def __init__(self, user, agent):
        self.user = user
        self.agent = agent
        self.user_log = []
        self.prev_agent_act = None

    def start(self):
        """Executes a dialog session by having the agent and the user take
        alternating turns.
        """
        # The agent starts the dialog
        agent_act = self.agent.start_dialog()
        user_act = UserAction(None, None)
        while not (user_act.type is UserActionType.CLOSE and
                   agent_act.type is AgentActionType.CLOSE):
            user_act = self.user.take_turn(agent_act)
            self._save_user_state_action(user_act)

            if agent_act.type is AgentActionType.CLOSE:
                break

            agent_act = self.agent.take_turn(user_act)

    def clear_user_log(self):
        """Purges the user log.
        """
        self.user_log[:] = []

    def ask_agent_to_start(self):
        """Makes the dialog agent start the dialog.

        Returns:
            AgentActionType: The type of action performed by the agent.
        """
        self.prev_agent_act = self.agent.start_dialog()
        return self.prev_agent_act.type

    def execute_one_step(self):
        """Executes one step of dialog by making the user act, followed
        by an action from the agent.

        Returns:
            UserActionType, AgentActionType: The type of action taken by the
                user, and it's response from the agent.
        """
        user_act = self.user.take_turn(self.prev_agent_act)
        self.prev_agent_act = self.agent.take_turn(user_act)
        return user_act.type, self.prev_agent_act.type

    def _save_user_state_action(self, user_action):
        """Appends the user's current state and action to the `user_log`.
        The state of the user is the agent's last action; other state
        attributes of UserState are ignored because they are not consequential
        to the user policy. The user action saved is the `type` of UserAction.

        Args:
            user_action (UserAction): Action taken by the user
        """
        # Only the `agent_act` attribute of `UserState` is consequential.
        state = self.user.state.agent_act.type
        action = user_action.type
        self.user_log.append((state, action))


###################################################################
# Sample usage to run a dialog session, or generate a dialog corpus
###################################################################

def generate_dialog_corpus(num_sessions):
    """Generates a dialog corpus by executing multiple sessions successively.

    Args:
        num_sessions (int, optional): Number of dialog sessions to be executed.
    """
    user = User(policy_type=UserPolicyType.handcrafted)
    agent = Agent()
    for _ in xrange(num_sessions):
        session = DialogSession(user, agent)
        session.start()
        print("----")
        session.clear_user_log()


def run_single_session():
    """Executes a single dialog session.
    """
    user = User(policy_type=UserPolicyType.handcrafted)
    agent = Agent()
    session = DialogSession(user, agent)
    session.start()
