"""Run a single dialog session."""

from agent.dialog_manager import DialogManager
from imitation_learning.dialog_session import DialogSession
from user.user import User
from utils.params import UserPolicyType


def run_single_session():
    """Executes a single dialog session.
    """
    user = User(UserPolicyType.handcrafted)
    agent = DialogManager()
    session = DialogSession(user, agent)
    session.start()


if __name__ == '__main__':
    run_single_session()
