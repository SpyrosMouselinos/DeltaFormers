import abc

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from .utils import inv_sherman_morrison


class UCB(abc.ABC):
    """Base class for UBC methods.
    """

    def __init__(self,
                 bandit,
                 reg_factor=1.0,
                 confidence_scaling_factor=-1.0,
                 delta=0.1,
                 train_every=1,
                 throttle=int(100),
                 mock_reset=False
                 ):
        # bandit object, contains features and generated rewards
        self.bandit = bandit
        # L2 regularization strength
        self.reg_factor = reg_factor
        # Confidence bound with probability 1-delta
        self.delta = delta
        # multiplier for the confidence bound (default is bandit reward noise std dev)
        if confidence_scaling_factor == -1.0:
            confidence_scaling_factor = bandit.noise_std
        self.confidence_scaling_factor = confidence_scaling_factor

        # train approximator only every few rounds
        self.train_every = train_every

        # throttle tqdm updates
        self.throttle = throttle
        self.mock_reset = mock_reset
        self.reset()

    def reset_upper_confidence_bounds(self):
        """Initialize upper confidence bounds and related quantities.
        """
        self.exploration_bonus = np.empty(self.bandit.n_arms)
        self.mu_hat = np.empty(self.bandit.n_arms)
        self.upper_confidence_bounds = np.ones(self.bandit.n_arms)

    def reset_actions(self):
        """Initialize cache of actions.
        """
        self.actions = []

    def reset_A_inv(self):
        """Initialize n_arms square matrices representing the inverses
        of exploration bonus matrices.
        """
        self.A_inv = np.array(
            [
                np.eye(self.approximator_dim) / self.reg_factor for _ in self.bandit.arms
            ]
        )

    def reset_grad_approx(self):
        """Initialize the gradient of the approximator w.r.t its parameters.
        """
        self.grad_approx = np.zeros((self.bandit.n_arms, self.approximator_dim))

    def sample_action(self):
        """Return the action to play based on current estimates
        """
        return np.argmax(self.upper_confidence_bounds).astype('int')

    @abc.abstractmethod
    def reset(self):
        """Initialize variables of interest.
        To be defined in children classes.
        """
        pass

    @property
    @abc.abstractmethod
    def approximator_dim(self):
        """Number of parameters used in the approximator.
        """
        pass



    @property
    @abc.abstractmethod
    def confidence_multiplier(self):
        """Multiplier for the confidence exploration bonus.
        To be defined in children classes.
        """
        pass


    def evaluate_output_gradient(self, features):
        """Compute output gradient of the approximator w.r.t its parameters.
        """
        pass

    @abc.abstractmethod
    def update_output_gradient(self):
        """Compute output gradient of the approximator w.r.t its parameters.
        """
        pass

    @abc.abstractmethod
    def train(self):
        """Update approximator.
        To be defined in children classes.
        """
        pass

    @abc.abstractmethod
    def predict(self):
        """Predict rewards based on an approximator.
        To be defined in children classes.
        """
        pass

    @abc.abstractmethod
    def evaluate(self, features):
        """Predict rewards based on an approximator.
        To be defined in children classes.
        """
        pass

    @abc.abstractmethod
    def save(self,postfix=''):
        pass

    def evaluate_confidence_bounds(self, features):
        """Update confidence bounds and related quantities for all arms.
        """
        self.evaluate_output_gradient(features)
        self.evaluate(features)
        # estimated combined bound for reward
        self.upper_confidence_bounds = self.mu_hat

    def update_confidence_bounds(self):
        """Update confidence bounds and related quantities for all arms.
        """
        self.update_output_gradient()

        # UCB exploration bonus
        self.exploration_bonus = np.array(
            [
                self.confidence_multiplier * np.sqrt(
                    np.dot(self.grad_approx[a], np.dot(self.A_inv[a], self.grad_approx[a].T))) for a in self.bandit.arms
            ]
        )

        # update reward prediction mu_hat
        self.predict()

        # estimated combined bound for reward
        self.upper_confidence_bounds = self.mu_hat + self.exploration_bonus

    def update_A_inv(self):
        self.A_inv[self.action] = inv_sherman_morrison(
            self.grad_approx[self.action],
            self.A_inv[self.action]
        )

    def run(self, epochs=10, save_every_epochs=1000, postfix=''):
        """Run an episode of bandit.
        """
        tqpostfix = {
            'Batches': 0.0,
            'Epochs': 0.0,
            'Regret': 0.0,
        }
        regret = []
        with tqdm(total=epochs * self.bandit.T, postfix=tqpostfix) as pbar:
            for epoch in range(epochs):
                batch_regret = np.zeros(self.bandit.T)
                for t in range(self.bandit.T):
                    self.update_confidence_bounds()
                    self.action = self.sample_action()
                    ### Calculate Regret ###
                    best_reward = max(self.bandit.rewards[t,:])
                    chosen_reward = self.bandit.rewards[t, self.action]
                    batch_regret[t] = best_reward - chosen_reward
                    if t % self.train_every == 0 and t > 0:
                        self.train()
                    self.update_A_inv()
                    # increment counter
                    self.iteration += 1
                    # get next batch
                self.bandit.reset()

                tqpostfix['Batches'] = (epoch + 1) * self.bandit.T
                tqpostfix['Epochs'] = epoch
                tqpostfix['Regret'] = batch_regret.mean()
                for j in range(self.bandit.T):
                    regret.append(batch_regret[j])
                pbar.set_postfix(tqpostfix)
                pbar.update(self.bandit.T)
                if epoch % save_every_epochs == 0 and epoch > 0:
                    self.save(postfix=postfix)
        plt.figure(figsize=(10,10))
        plt.title(f"Regret VS Epochs for LinUCB @ {self.bandit.augmentation_strength}")
        plt.plot(regret, 'b')
        plt.ylim([0, 1])
        plt.savefig(f'./regret_plot_{self.bandit.augmentation_strength}.png')
        plt.close()


    def test(self, features):
        """
        Evaluate an example
        """
        self.evaluate_confidence_bounds(features)
        self.action = self.sample_action()
        return self.upper_confidence_bounds, self.mu_hat, self.action
