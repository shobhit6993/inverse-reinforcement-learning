"""Run a single dialog session."""

from imitation_learning.dialog_session import DialogSession
from utils.params import UserPolicyType


def run_single_session():
    """Executes a single dialog session.
    """
    session = DialogSession(UserPolicyType.handcrafted)
    session.start()
    print session.user_log


if __name__ == '__main__':
    run_single_session()
