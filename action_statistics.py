import sys
from agent.agent import Agent
from imitation_learning.dialog_session import DialogSession
from user.user import User
from utils.params import UserPolicyType, UserActionType, NUM_SESSIONS_FE
from run_best_user_simulation import load_best_user_simulation
from simulation.mixed_user_simulation import QpMixedUserSimulation, GibbsMixedUserSimulation


def run_single_session(user):
    """Executes a single dialog session.
    """
    agent = Agent()
    user.reset(reset_policy=False)
    session = DialogSession(user, agent)
    session.start()
    return session.user_log


def expert(num_sessions):
    freq = {}
    for action in UserActionType:
        freq[action] = 0

    user = User(policy_type=UserPolicyType.handcrafted)
    for _ in xrange(num_sessions):
        user_log = run_single_session(user)
        for _, action in user_log:
            freq[action] += 1

    for action in freq.keys():
        freq[action] /= (1. * num_sessions)
    print freq


def best(num_sessions, filepath):
    freq = {}
    for action in UserActionType:
        freq[action] = 0

    user = load_best_user_simulation(filepath)
    for _ in xrange(num_sessions):
        user_log = run_single_session(user)
        for _, action in user_log:
            freq[action] += 1

    for action in freq.keys():
        freq[action] /= (1. * num_sessions)
    print freq


def softmax(num_sessions, filepath):
    simulation = GibbsMixedUserSimulation(filepath)
    print simulation.temp(num_sessions)


def qp(num_sessions, filepath):
    real_user = User(policy_type=UserPolicyType.handcrafted)
    simulation = QpMixedUserSimulation(filepath, real_user)
    simulation.solve_qp()
    print simulation.temp(num_sessions)


if __name__ == '__main__':
    expert(NUM_SESSIONS_FE)
    raw_input()
    best(NUM_SESSIONS_FE, sys.argv[1])
    raw_input()
    softmax(NUM_SESSIONS_FE, sys.argv[1])
    raw_input()
    qp(NUM_SESSIONS_FE, sys.argv[1])
    raw_input()
