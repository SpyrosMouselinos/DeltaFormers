import os
import pickle

import numpy as np
from numba import jit

from .ucb import UCB


@jit(nopython=True)
def calc_confidence_multiplier(confidence_scaling_factor, approximator_dim, iteration, bound_features,
                               reg_factor, delta):
    return confidence_scaling_factor * np.sqrt(approximator_dim * np.log(
        1 + iteration * bound_features ** 2 / (reg_factor * approximator_dim)) + 2 * np.log(1 / delta))


class LinUCB(UCB):
    """Linear UCB.
    """

    def __init__(self,
                 bandit,
                 reg_factor=1.0,
                 delta=0.01,
                 bound_theta=1.0,
                 confidence_scaling_factor=0.0,
                 save_path=None,
                 load_from=None,
                 guide_for=0,
                 ):
        self.iteration = 0
        self.save_path = save_path
        if load_from is None:
            # range of the linear predictors
            self.bound_theta = bound_theta

            # maximum L2 norm for the features across all arms and all rounds
            self.bound_features = np.max(np.linalg.norm(bandit.features, ord=2, axis=-1))

            super().__init__(bandit,
                             reg_factor=reg_factor,
                             confidence_scaling_factor=confidence_scaling_factor,
                             delta=delta,
                             mock_reset=False,
                             guide_for=guide_for
                             )
        else:
            state_dict = self.load(load_from)
            self.bound_theta = state_dict['bound_theta']
            self.bound_features = np.max(np.linalg.norm(bandit.features, ord=2, axis=-1))
            super().__init__(bandit,
                             reg_factor=reg_factor,
                             confidence_scaling_factor=confidence_scaling_factor,
                             delta=delta,
                             mock_reset=True,

                             )
            self.reg_factor = state_dict['reg_factor']
            self.delta = state_dict['delta']
            self.bound_theta = state_dict['bound_theta']
            self.confidence_scaling_factor = state_dict['confidence_scaling_factor']
            self.A_inv = state_dict['A_inv']
            self.theta = state_dict['theta']
            self.b = state_dict['b']
            self.iteration = state_dict['iteration']

    def save(self, postfix=''):
        if self.save_path is None:
            print("Save path is empty...saving here\n")
            self.save_path = './'
        state_dict = {
            'reg_factor': self.reg_factor,
            'delta': self.delta,
            'bound_theta': self.bound_theta,
            'confidence_scaling_factor': self.confidence_scaling_factor,
            'A_inv': self.A_inv,
            'theta': self.theta,
            'b': self.b,
            'iteration': self.iteration,
        }
        with open(self.save_path + f'/linucb_model_{postfix}.pt', 'wb') as f:
            pickle.dump(state_dict, f)

        if os.path.exists(self.save_path + f'/linucb_model_{postfix}.pt'):
            print("Model Saved Succesfully!")
        return

    def load(self, path):
        with open(path, 'rb') as f:
            state_dict = pickle.load(f)
        return state_dict

    @property
    def approximator_dim(self):
        """Number of parameters used in the approximator.
        """
        return self.bandit.n_features

    def update_output_gradient(self):
        """For linear approximators, simply returns the features.
        Gradient for each arm here is the same. = Input
        """
        self.grad_approx = self.bandit.features[self.iteration % self.bandit.T]

    def evaluate_output_gradient(self, features):
        """For linear approximators, simply returns the features.
        """
        self.grad_approx = features[0]

    def reset(self):
        """Return the internal estimates.
            Initialize SINGLE PREDICTOR AND SINGLE B.
            REMEMBER TO USE ONLY THE FIRST A_INV
        """
        self.reset_upper_confidence_bounds()
        self.reset_actions()
        self.reset_grad_approx()
        if not self.mock_reset:
            self.reset_A_inv()
            self.theta = np.random.uniform(-1, 1, self.bandit.n_features) * self.bound_theta
            self.b = np.zeros(self.bandit.n_features)

    @property
    def confidence_multiplier(self):
        """Confidence interval multiplier.
        """
        return calc_confidence_multiplier(self.confidence_scaling_factor, self.approximator_dim, self.iteration,
                                               self.bound_features, self.reg_factor, self.delta)

    def predict(self):
        """Predict reward.
        """
        self.mu_hat = np.array(
            [
                np.dot(self.bandit.features[self.iteration % self.bandit.T, a], self.theta.T) for a in self.bandit.arms
            ]
        )

    def train(self):
        """Update linear predictor theta.
        """
        self.b += self.bandit.features[self.iteration % self.bandit.T, self.action] * self.bandit.rewards[
            self.iteration % self.bandit.T, self.action]

        self.theta = np.matmul(self.A_inv, self.b)

    def evaluate(self, features):
        self.mu_hat = np.array(
            [
                np.dot(features[0, a], self.theta.T) for a in self.bandit.arms
            ]
        )
