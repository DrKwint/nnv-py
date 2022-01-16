from nnv_py.nnv_py import PyStarSet  # pylint: disable=no-name-in-module
import numpy as np


class StarSet:
    def __init__(self, dnn):
        self.starset = PyStarSet(dnn.dnn)

    def min_adversarial_delta(self, x):
        pass

    def min_adversarial_delta_targeted(self, x, target_y):
        return self.starset.minimal_norm_targeted_attack_delta(x, target_y)