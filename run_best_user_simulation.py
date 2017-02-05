import sys

from agent.agent import Agent
from imitation_learning.dialog_session import DialogSession
from simulation.best_user_simulation import BestUserSimulation
from utils.params import NUM_SESSIONS_FE
from utils.utils import collect_statistics


def load_best_user_simulation(filepath):
    """Loads the best user-simulation from a dump of user-simulations leant
    through IRL.

    Args:
        filepath (string): Path of the user-simulations dump

    Returns:
        BestUserSimulation: The user-simulation whose feature expectation is
            closest to that of the expert user.
    """
    simulation = BestUserSimulation()
    simulation.find_best_simulation(filepath)
    return simulation


if __name__ == '__main__':
    user = load_best_user_simulation(sys.argv[1])
    agent = Agent()
    collect_statistics(user, agent, DialogSession, NUM_SESSIONS_FE)
