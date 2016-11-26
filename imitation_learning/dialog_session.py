"""Dialog session."""

from agent.dialog_manager import DialogManager
from user.user import User
from user.user_action import UserAction
from utils.params import AgentActionType, NUM_SESSIONS_IN_CORPUS
from utils.params import UserActionType, UserPolicyType


class DialogSession:
    """Class for a single dialog session.

    Attributes:
        agent (DialogManager): The dialog agent, also called Dialog Manager.
        user (User): The user participating in the dialog.
        user_log (list of tuples): Log of (state, action) pairs that the user
            underwent in this dialog session in the form of a list of tuples of
            form (AgentActionType, UserActionType).
    """

    def __init__(self, user_policy_type):
        """Class constructor

        Args:
            user_policy_type (UserPolicyType): Type of user policy.
        """
        self.user = User(user_policy_type)
        self.agent = DialogManager()
        self.user_log = []

    def start(self):
        """Executes a dialog session by having the agent and the user take
        alternating turns.
        """
        # The agent starts the dialog
        system_act = self.agent.start_dialog()
        user_act = UserAction(None, None)
        while not (user_act.type is UserActionType.CLOSE and
                   system_act.type is AgentActionType.CLOSE):
            user_act = self.user.take_turn(system_act)
            self._save_user_state_action(user_act)

            if system_act.type is AgentActionType.CLOSE:
                break

            system_act = self.agent.take_turn(user_act)

    def _save_user_state_action(self, user_action):
        """Appends the user's current state and action to the `user_log`.
        The state of the user is the agent's last action; other state
        attributes of UserState are ignored because they are not consequential
        to the user policy. The user action saved is the `type` of UserAction.

        Args:
            user_action (UserAction): Action taken by the user
        """
        # Only the `system_act` attribute of `UserState` is consequential.
        state = self.user.state.system_act.type
        action = user_action.type
        self.user_log.append((state, action))


###################################################################
# Sample usage to run a dialog session, or generate a dialog corpus
###################################################################

def generate_dialog_corpus(num_sessions=NUM_SESSIONS_IN_CORPUS):
    """Generates a dialog corpus by executing multiple sessions successively.

    Args:
        num_sessions (int, optional): Number of dialog sessions to be executed.
    """
    for _ in xrange(num_sessions):
        session = DialogSession(UserPolicyType.handcrafted)
        session.start()
        print("----")


def run_single_session():
    """Executes a single dialog session.
    """
    session = DialogSession(UserPolicyType.handcrafted)
    session.start()
