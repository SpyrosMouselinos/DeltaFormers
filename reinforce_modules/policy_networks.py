import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import MultivariateNormal


def _print(something):
    print(something, flush=True)
    return


class PolicyNet(nn.Module):
    def __init__(self, input_size, dropout=0.0):
        super(PolicyNet, self).__init__()
        self.input_size = input_size
        # Pre-Processing #
        self.fc1 = nn.Linear(input_size, input_size)
        self.fc1_d = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(input_size, input_size // 2)
        self.fc2_d = nn.Dropout(p=dropout)
        # Output #
        self.mu = nn.Linear(input_size // 2, 20)
        # self.logsigma_diag = nn.Linear(input_size // 2, 20)
        self.sigma_diag = nn.Linear(input_size // 2, 20)

    def forward(self, x):
        x = self.fc1_d(F.relu(self.fc1(x), inplace=True))
        x = self.fc2_d(F.relu(self.fc2(x), inplace=True))
        # Means are limited to [-0.3,+0.3] Range
        mu = 0.3 * torch.tanh(self.mu(x))
        # Stds are limited to (0,1] range -> Log(Std) is limited to (-inf, 0]
        sigma_diag = torch.sigmoid(self.sigma_diag(x)) + 1e-2
        return mu, sigma_diag

    def save(self, path):
        torch.save(self.state_dict(), path)
        if os.path.exists(path):
            return True
        else:
            return False

    def load(self, path):
        self.load_state_dict(torch.load(path))
        return


class Re1nforceTrainer:
    def __init__(self, model, game, dataloader, device='cuda', lr=0.001, train_duration=10, batch_size=32,
                 batch_tolerance=1, batch_reset=100):
        self.model = model
        self.game = game
        self.dataloader = dataloader
        self.dataloader_iter = iter(self.dataloader)
        self.device = device
        self.training_duration = train_duration
        self.batch_size = batch_size
        self.lr = lr
        self.batch_tolerance = batch_tolerance
        self.batch_reset = batch_reset

    def train(self, log_every=100, save_every=10_000):
        self.model = self.model.to(self.device)
        self.model = self.model.train()
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)

        self.model.zero_grad()

        accuracy_drop = []
        batch_idx = 0
        epochs_passed = 0
        # marked_batches = {}
        while batch_idx < self.training_duration:
            # Reset all marked batches !#
            # if epochs_passed % self.batch_reset == 0 and epochs_passed > 0:
            #     marked_batches = {}

            # Dont use marked batches !#
            # if batch_idx % len(self.dataloader) in marked_batches:
            #     if marked_batches[batch_idx % len(self.dataloader)] >= self.batch_tolerance:
            #         # Burn Batch #
            #         _print(f"Burning Batch {batch_idx} | equivalent to {batch_idx % len(self.dataloader)}...")
            #         _ = next(self.dataloader_iter)
            #         batch_idx += 1
            #         continue
            try:
                features = self.game.extract_features(self.dataloader_iter)
            except StopIteration:
                del self.dataloader_iter
                self.dataloader_iter = iter(self.dataloader)
                features = self.game.extract_features(self.dataloader_iter)
                epochs_passed += 1

            # Forward Pass #
            mu, sigma_diag = self.model(features)
            cov_mat = torch.diag_embed(sigma_diag)
            m = MultivariateNormal(loc=mu, covariance_matrix=cov_mat)
            mixed_actions = m.sample()

            # Calc Reward #
            rewards, confusion_rewards, _, _, _ = self.game.get_rewards(mixed_actions)
            loss = -m.log_prob(mixed_actions) * torch.FloatTensor(rewards).squeeze(1).to(self.device)
            loss = loss.mean()
            # Null Grad #
            optimizer.zero_grad()
            loss.backward()
            # Clip Norms of 10 and higher #
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
            optimizer.step()
            batch_accuracy = 100 * (confusion_rewards.squeeze(1).mean())
            accuracy_drop.append(batch_accuracy)
            if batch_idx % log_every == 0 and batch_idx > 0:
                _print(
                    f"REINFORCE  {batch_idx} / {self.training_duration} | Accuracy Dropped By: {np.array(accuracy_drop[-10_000:]).mean()}%")
            if batch_idx % save_every == 0 and batch_idx > 0:
                self.model.save('./results/experiment_reinforce/model_reinforce.pt')

            # if batch_accuracy == 0.0:
            #     # If batch has 0 accuracy mark it  MAKE SURE LOADER DOES NOT SHUFFLE!#
            #     if batch_idx % len(self.dataloader) not in marked_batches:
            #         marked_batches.update({batch_idx % len(self.dataloader): 1})
            #     else:
            #         marked_batches[batch_idx % len(self.dataloader)] += 1
            #     _print(f"Found {len(marked_batches)} bad batches so far")

            batch_idx += 1
        self.model.save('./results/experiment_reinforce/model_reinforce.pt')
        plt.figure(figsize=(10, 10))
        plt.title('REINFORCE Accuracy Drop Progress')
        plt.plot(accuracy_drop, 'b')
        plt.ylabel("Accuracy Drop on 96% State Transformer")
        plt.xlabel("Training Iterations")
        plt.savefig('./results/experiment_reinforce/progress.png')
        plt.show()
        plt.close()
