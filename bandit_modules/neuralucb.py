import numpy as np
import torch
import os
import pickle
from numba import jit
import torch.nn as nn
from .ucb import UCB
from .utils import Model

@jit(nopython=True)
def calc_confidence_multiplier(confidence_scaling_factor, approximator_dim, iteration, bound_features,
                               reg_factor, delta):
    return confidence_scaling_factor * np.sqrt(approximator_dim * np.log(
        1 + iteration * bound_features ** 2 / (reg_factor * approximator_dim)) + 2 * np.log(1 / delta))



class NeuralUCB(UCB):
    """Neural UCB.
    """
    def __init__(self,
                 bandit,
                 hidden_size=20,
                 n_layers=2,
                 reg_factor=1.0,
                 delta=0.01,
                 confidence_scaling_factor=-1.0,
                 training_window=100,
                 p=0.0,
                 learning_rate=0.01,
                 epochs=1,
                 train_every=1,
                 save_path=None,
                 load_from=None,
                 guide_for=0
                 ):
        self.iteration = 0
        self.save_path = save_path
        self.rhs = np.sqrt(hidden_size)
        if load_from is None:
            # hidden size of the NN layers
            self.hidden_size = hidden_size
            # number of layers
            self.n_layers = n_layers

            # number of rewards in the training buffer
            self.training_window = training_window

            # NN parameters
            self.learning_rate = learning_rate
            self.epochs = epochs

            self.device = 'cpu'#torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # dropout rate
            self.p = p

            # neural network
            self.model = Model(input_size=bandit.n_features,
                               hidden_size=self.hidden_size,
                               n_layers=self.n_layers,
                               p=self.p
                               ).to(self.device)


            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

            # maximum L2 norm for the features across all arms and all rounds
            self.bound_features = np.max(np.linalg.norm(bandit.features, ord=2, axis=-1))

            super().__init__(bandit,
                             reg_factor=reg_factor,
                             confidence_scaling_factor=confidence_scaling_factor,
                             delta=delta,
                             train_every=train_every,
                             guide_for=guide_for
                             )
        else:
            raise NotImplementedError

    def save(self, postfix=''):
        return

    @property
    def approximator_dim(self):
        """Sum of the dimensions of all trainable layers in the network.
        """
        return sum(w.numel() for w in self.model.parameters() if w.requires_grad)



    def update_output_gradient(self):
        """Get gradient of network prediction w.r.t network weights.
            Gradient for each arm.
        """
        for a in self.bandit.arms:
            x = torch.FloatTensor(
                self.bandit.features[self.iteration % self.bandit.T, a].reshape(1, -1)
            ).to(self.device)

            self.model.zero_grad()
            y = self.model(x)
            y.backward()

            self.grad_approx[a] = torch.cat(
                [w.grad.detach().cpu().flatten() / self.rhs for w in self.model.parameters() if w.requires_grad]
            ).numpy()

    def evaluate_output_gradient(self, features):
        """For linear approximators, simply returns the features.
        """
        for a in self.bandit.arms:
            x = torch.FloatTensor(
                features[0, a].reshape(1, -1)
            ).to(self.device)

            self.model.zero_grad()
            y = self.model(x)
            y.backward()

            self.grad_approx[a] = torch.cat(
                [w.grad.detach().cpu().flatten() / self.rhs for w in self.model.parameters() if w.requires_grad]
            ).numpy()

    def reset(self):
        """Return the internal estimates.
            Initialize SINGLE PREDICTOR NN.
            REMEMBER TO USE ONLY THE FIRST A_INV
        """
        self.reset_upper_confidence_bounds()
        self.reset_actions()
        self.reset_A_inv()
        self.reset_grad_approx()



    @property
    def confidence_multiplier(self):
        """Confidence interval multiplier.
        """
        return calc_confidence_multiplier(self.confidence_scaling_factor, self.approximator_dim, self.iteration,
                                               self.bound_features, self.reg_factor, self.delta)


    def train(self):
        """
            Train neural approximator.
        """
        iterations_so_far = range(np.max([0, (self.iteration % self.bandit.T) - self.training_window]), (self.iteration % self.bandit.T)+1)
        actions_so_far = self.actions[np.max([0, (self.iteration % self.bandit.T) - self.training_window]):(self.iteration % self.bandit.T)+1]

        x_train = torch.FloatTensor(self.bandit.features[iterations_so_far, actions_so_far]).to(self.device)
        y_train = torch.FloatTensor(self.bandit.rewards[iterations_so_far, actions_so_far]).squeeze().to(self.device)

        # train mode
        self.model.train()
        for _ in range(self.epochs):
            y_pred = self.model.forward(x_train).squeeze()
            loss = nn.MSELoss()(y_train, y_pred)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def predict(self):
        """Predict reward.
        """
        # eval mode
        self.model.eval()
        self.mu_hat = self.model.forward(
            torch.FloatTensor(self.bandit.features[self.iteration % self.bandit.T]).to(self.device)
        ).detach().squeeze().cpu().numpy()


    def evaluate(self, features):
        # eval mode
        self.model.eval()
        self.mu_hat = self.model.forward(
            torch.FloatTensor(features[0]).to(self.device)
        ).detach().squeeze().cpu().numpy()