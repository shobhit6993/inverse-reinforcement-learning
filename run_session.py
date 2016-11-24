from agent.dialog_manager import DialogManager
from utils.params import AgentActionType, UserActionType, NUM_SESSIONS_IN_CORPUS
from user.real_user import RealUser
from user.user_action import UserAction


class DialogSession:
    def __init__(self):
        self.user = RealUser()
        self.agent = DialogManager()

    def start(self):
        # The agent starts the dialog
        system_act = self.agent.start_dialog()
        user_act = UserAction(None, None)
        while not (user_act.type is UserActionType.CLOSE and
                   system_act.type is AgentActionType.CLOSE):
            user_act = self.user.take_turn(system_act)

            if system_act.type is AgentActionType.CLOSE:
                break

            system_act = self.agent.take_turn(user_act)
            # raw_input()


def generate_dialog_corpus(num_sessions=NUM_SESSIONS_IN_CORPUS):
    for _ in xrange(num_sessions):
        session = DialogSession()
        session.start()
        print("----")


def main():
    session = DialogSession()
    session.start()


if __name__ == '__main__':
    # generate_dialog_corpus()
    main()
