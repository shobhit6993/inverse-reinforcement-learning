from copy import deepcopy
import pickle
import numpy as np

from user_simulation import UserSimulation


class BestUserSimulation(UserSimulation):
    def find_best_simulation(self, filepath):
        user_simulations = self._load_user_simulations(filepath)
        distances = np.array([sim.distance_to_expert
                              for sim in user_simulations])
        max_index = np.argmin(distances)
        best_simulation = user_simulations[max_index]

        self.policy = deepcopy(best_simulation.policy)
        self.distance_to_expert = deepcopy(best_simulation.distance_to_expert)

    def _load_user_simulations(self, filepath):
        with open(filepath, "r") as fin:
            user_simulations = pickle.load(fin)
            return user_simulations
