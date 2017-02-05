import sys

from agent.agent import Agent
from imitation_learning.dialog_session import DialogSession
from simulation.mixed_user_simulation import QpMixedUserSimulation, GibbsMixedUserSimulation
from user.user import User
from utils.params import NUM_SESSIONS_FE, UserPolicyType
from utils.utils import collect_statistics


def load_qp_mixed_user_simulation(filepath):
    """Loads the best user-simulation using QP formulation from a dump of
    user-simulations leant through IRL.

    Args:
        filepath (string): Path of the user-simulations dump

    Returns:
        BestUserSimulation: The user-simulation whose feature expectation is
            closest to that of the expert user.
    """
    real_user = User(policy_type=UserPolicyType.handcrafted)
    simulation = QpMixedUserSimulation(filepath, real_user)
    simulation.solve_qp()
    print simulation.mixture_weights
    simulation.collect_statistics(NUM_SESSIONS_FE)


def load_gibbs_mixed_user_simulation(filepath):
    """Loads the best user-simulation using Gibbs mixing from a dump of
    user-simulations leant through IRL.

    Args:
        filepath (string): Path of the user-simulations dump

    Returns:
        BestUserSimulation: The user-simulation whose feature expectation is
            closest to that of the expert user.
    """
    simulation = GibbsMixedUserSimulation(filepath)
    simulation.collect_statistics(NUM_SESSIONS_FE)


if __name__ == '__main__':
    load_gibbs_mixed_user_simulation(sys.argv[1])
    raw_input()
    load_qp_mixed_user_simulation(sys.argv[1])
