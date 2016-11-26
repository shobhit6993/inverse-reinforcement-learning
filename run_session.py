"""Run dialog session."""

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
    """

    def __init__(self, user_policy_type):
        """Class constructor

        Args:
            user_policy_type (UserPolicyType): Type of user policy.
        """
        self.user = User(user_policy_type)
        self.agent = DialogManager()

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

            if system_act.type is AgentActionType.CLOSE:
                break

            system_act = self.agent.take_turn(user_act)


def generate_dialog_corpus(num_sessions=NUM_SESSIONS_IN_CORPUS):
    """Generates a dialog corpus by executing multiple sessions successively.

    Args:
        num_sessions (int, optional): Number of dialog sessions to be executed.
    """
    for _ in xrange(num_sessions):
        session = DialogSession(UserPolicyType.handcrafted)
        session.start()
        print("----")


def run_single_seesion():
    """Executes a single dialog session.
    """
    session = DialogSession(UserPolicyType.handcrafted)
    session.start()


if __name__ == '__main__':
    # generate_dialog_corpus()
    run_single_seesion()
