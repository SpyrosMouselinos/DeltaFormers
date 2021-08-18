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


class QuestionReintroduce(nn.Module):
    def __init__(self, input_size, hidden_size, reverse_input=False):
        super(QuestionReintroduce, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reverse_input = reverse_input
        self.embeddings = nn.Embedding(num_embeddings=93, embedding_dim=self.input_size, padding_idx=0)
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True)

    def forward(self, x):
        if self.reverse_input:
            x = torch.flip(x, [1])
        emb_x = self.embeddings(x)
        aggregated = self.lstm(emb_x)
        return aggregated

    def save(self, path):
        torch.save(self.state_dict(), path)
        if os.path.exists(path):
            return True
        else:
            return False

    def load(self, path):
        self.load_state_dict(torch.load(path))
        return


class FFNet(nn.Module):
    def __init__(self, input_size, dropout=0.0):
        super(FFNet, self).__init__()
        self.input_size = input_size
        # Pre-Processing #
        self.fc1 = nn.Linear(input_size, input_size)
        self.fc1_d = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(input_size, input_size)
        self.fc2_d = nn.Dropout(p=dropout)
        self.fc3 = nn.Linear(input_size, input_size // 2)
        # self.fc3_d = nn.Dropout(p=dropout)
        # self.fc4 = nn.Linear(input_size, input_size // 2)
        # self.fc4_d = nn.Dropout(p=dropout)
        # Output #
        self.mu = nn.Linear(input_size // 2, 20)
        # self.logsigma_diag = nn.Linear(input_size // 2, 20)
        self.sigma_diag = nn.Linear(input_size // 2, 20)

    def forward(self, x):
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        x = F.relu(self.fc3(x), inplace=True)
        # x = self.fc4_d(F.relu(self.fc4(x), inplace=True))
        # Means are limited to [-0.3,+0.3] Range
        mu = 0.3 * torch.tanh(self.mu(x))
        # Stds are limited to (0,1] range -> Log(Std) is limited to (-inf, 0]
        sigma_diag = 0.00001 * torch.sigmoid(self.sigma_diag(x)) + 1e-5
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


class PolicyNet(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.0, reverse_input=False):
        super(PolicyNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reverse_input = reverse_input
        self.dropout = dropout
        self.question_model = QuestionReintroduce(input_size, hidden_size, reverse_input)
        self.final_model = FFNet(hidden_size + 128, dropout)

    def forward(self, x, q):
        _, (q_summary, _) = self.question_model(q)
        q_summary = torch.squeeze(q_summary, 0)
        mix = torch.cat([x, q_summary], 1)
        mu, sigma_diag = self.final_model(mix)
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
        epoch_accuracy_drop = 0
        epoch_accuracy_drop_history = []
        # marked_batches = {}
        while batch_idx < self.training_duration:
            try:
                features, org_data = self.game.extract_features(self.dataloader_iter)
            except StopIteration:
                del self.dataloader_iter
                self.dataloader_iter = iter(self.dataloader)
                features, org_data = self.game.extract_features(self.dataloader_iter)
                _print(
                    f"REINFORCE  Epoch {epochs_passed} | Epoch Accuracy Drop: {epoch_accuracy_drop / len(self.dataloader)}%")
                epochs_passed += 1
                epoch_accuracy_drop_history.append(epoch_accuracy_drop / len(self.dataloader))
                epoch_accuracy_drop = 0

            # Forward Pass #
            mu, sigma_diag = self.model(features, org_data['question'])
            cov_mat = torch.diag_embed(sigma_diag)
            m = MultivariateNormal(loc=mu, covariance_matrix=cov_mat)
            mixed_actions = m.sample()

            # Calc Reward #
            rewards, confusion_rewards, _, _, _ = self.game.get_rewards(mixed_actions)
            loss = -m.log_prob(mixed_actions) * torch.FloatTensor(rewards).squeeze(1).to(self.device)
            #loss = torch.FloatTensor(rewards).squeeze(1).to(self.device)
            loss = loss.mean()
            # Null Grad #
            optimizer.zero_grad()
            loss.backward()
            # Clip Norms of 10 and higher #
            if batch_idx % log_every == 0 and batch_idx > 0:
                total_norm = 0
                for p in self.model.parameters():
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1. / 2)
                _print(f"Gradient Norm is {total_norm}")
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2)
            optimizer.step()
            batch_accuracy = 100 * (confusion_rewards.squeeze(1).mean())
            accuracy_drop.append(batch_accuracy)
            epoch_accuracy_drop += batch_accuracy
            if batch_idx % log_every == 0 and batch_idx > 0:
                _print(
                    f"REINFORCE  {batch_idx} / {self.training_duration} | Accuracy Dropped By: {np.array(accuracy_drop).mean()}%")
            if batch_idx % save_every == 0 and batch_idx > 0:
                self.model.save('./results/experiment_reinforce/model_reinforce.pt')
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

        plt.figure(figsize=(10, 10))
        plt.title('REINFORCE Epoch Accuracy Drop Progress')
        plt.plot(epoch_accuracy_drop_history, 'b')
        plt.ylabel("Accuracy Drop on 96% State Transformer")
        plt.xlabel("Epochs")
        plt.savefig('./results/experiment_reinforce/epoch_progress.png')
        plt.show()
        plt.close()
