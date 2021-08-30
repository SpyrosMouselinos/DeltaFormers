import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.distributions import MultivariateNormal


def kwarg_dict_to_device(data_obj, device):
    cpy = {}
    for key, _ in data_obj.items():
        cpy[key] = data_obj[key].to(device)
    return cpy


def _print(something):
    print(something, flush=True)
    return


def accuracy_metric(y_pred, y_true):
    acc = (y_pred.argmax(1) == y_true).float().detach().cpu().numpy()
    return float(100 * acc.sum() / len(acc))


class EmptyNetwork(nn.Module):
    def __init__(self):
        super(EmptyNetwork, self).__init__()

    def forward(self, x):
        return x


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
        self.fc1 = nn.Linear(input_size, input_size // 2)
        self.fc2 = nn.Linear(input_size // 2, input_size // 2)
        self.fc3 = nn.Linear(input_size // 2, input_size // 2)

        # How many are there?
        self.hmat = nn.Linear(input_size // 2, 9)  # 2 up to 10 = 9
        # Output #
        self.mu = nn.Linear(input_size // 2, 20)
        # self.logsigma_diag = nn.Linear(input_size // 2, 20)
        # self.sigma_diag = nn.Linear(input_size // 4, 20)

    def forward(self, x):
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        x = F.relu(self.fc3(x), inplace=True)
        # x = self.fc4_d(F.relu(self.fc4(x), inplace=True))
        # Means are limited to [-0.3,+0.3] Range
        mu = self.mu(x)
        hmat = self.hmat(x)
        # mu = mu / torch.norm(mu, p=2, dim=1, keepdim=True)
        # Stds are limited to (0,1] range -> Log(Std) is limited to (-inf, 0]
        # sigma_diag = 0.00001 * torch.sigmoid(self.sigma_diag(x)) + 1e-5
        sigma_diag = 0.001 * torch.ones_like(mu)
        return mu, sigma_diag, hmat

    def save(self, path):
        torch.save(self.state_dict(), path)
        if os.path.exists(path):
            return True
        else:
            return False

    def load(self, path):
        self.load_state_dict(torch.load(path))
        return


class FeatureConvNet(nn.Module):
    def __init__(self):
        super(FeatureConvNet, self).__init__()
        self.resnet_base = self.__build_resnet_base__(3, True)
        self.conv1 = nn.Conv2d(256, 4, (3, 3))
        self.flat = nn.Flatten()
        self.reduce = nn.Linear(4 * 6 * 6, 128)

    def __build_resnet_base__(self, stage=3, frozen=True):
        whole_cnn = getattr(torchvision.models, 'resnet34')(pretrained=True)
        layers = [
            whole_cnn.conv1,
            whole_cnn.bn1,
            whole_cnn.relu,
            whole_cnn.maxpool,
        ]
        for i in range(stage):
            name = 'layer%d' % (i + 1)
            layers.append(getattr(whole_cnn, name))
        cnn = torch.nn.Sequential(*layers)
        if frozen:
            cnn.eval()
        else:
            cnn.train()
        return cnn

    def forward(self, image):
        x = image
        x = self.resnet_base(x)
        x = F.relu(self.conv1(x))
        x = self.flat(x)
        x = self.reduce(x)
        return x

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
        # x = x / torch.norm(x, 2, 1, True)
        # q_summary = self.layer_norm_q(q_summary)
        mix = torch.cat([x, q_summary], 1)
        mu, sigma_diag, hmat = self.final_model(mix)
        return mu, sigma_diag, hmat

    def save(self, path):
        torch.save(self.state_dict(), path)
        if os.path.exists(path):
            return True
        else:
            return False

    def load(self, path):
        self.load_state_dict(torch.load(path))
        return


class ImageNetPolicyNet(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.0, reverse_input=False):
        super(ImageNetPolicyNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reverse_input = reverse_input
        self.dropout = dropout
        self.image_preprocess_model = FeatureConvNet()
        self.question_model = QuestionReintroduce(input_size, hidden_size, reverse_input)
        self.final_model = FFNet(hidden_size + 128, dropout)

    def forward(self, x, q):
        _, (q_summary, _) = self.question_model(q)
        q_summary = torch.squeeze(q_summary, 0)
        # x = x / torch.norm(x, 2, 1, True)
        # q_summary = self.layer_norm_q(q_summary)

        x = self.image_preprocess_model(x)
        mix = torch.cat([x, q_summary], 1)
        mu, sigma_diag, hmat = self.final_model(mix)
        return mu, sigma_diag, hmat

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
        acc_loss = torch.nn.CrossEntropyLoss()
        accuracy_drop = []
        batch_idx = 0
        epochs_passed = 0
        epoch_accuracy_drop = 0
        epoch_accuracy_drop_history = []
        # marked_batches = {}
        while batch_idx < self.training_duration:
            try:
                features, org_data, _ = self.game.extract_features(self.dataloader_iter)
            except StopIteration:
                del self.dataloader_iter
                self.dataloader_iter = iter(self.dataloader)
                features, org_data, _ = self.game.extract_features(self.dataloader_iter)
                _print(
                    f"REINFORCE  Epoch {epochs_passed} | Epoch Accuracy Drop: {epoch_accuracy_drop / len(self.dataloader)}%")
                epochs_passed += 1
                epoch_accuracy_drop_history.append(epoch_accuracy_drop / len(self.dataloader))
                epoch_accuracy_drop = 0

            # Forward Pass #
            mu, sigma_diag, hmat = self.model(features, org_data['question'])
            cov_mat = torch.diag_embed(sigma_diag)
            m = MultivariateNormal(loc=mu, covariance_matrix=cov_mat)
            mixed_actions = m.sample().clamp(-0.3, 0.3)

            # # Calc Reward #
            rewards, confusion_rewards, change_rewards, fail_rewards, invalid_scene_rewards, scene, predictions_after = self.game.get_rewards(
                mixed_actions)
            loss = -m.log_prob(mixed_actions) / 50 * torch.FloatTensor(rewards).squeeze(1).to(self.device)
            loss = loss.mean()

            _x = mu[:, :10]
            _y = mu[:, 10:]
            # # # Magnitude Loss #
            # mask_1 = 1.0 * org_data['types'][:, :10].eq(1)  # Normal mask
            # mag_loss_over_x = F.relu(_x - 0.5)
            # mag_loss_under_x = F.relu(-1 * (_x + 0.5))
            # mag_loss_x = mag_loss_over_x + mag_loss_under_x
            # mag_loss_x = mag_loss_x * mask_1
            # mag_loss_over_y = F.relu(_y - 0.5)
            # mag_loss_under_y = F.relu(-1 * (_y + 0.5))
            # mag_loss_y = mag_loss_over_y + mag_loss_under_y
            # mag_loss_y = mag_loss_y * mask_1
            # mag_loss = mag_loss_x.mean() + mag_loss_y.mean()

            # # Auxilliary Identification Loss #
            mask_0 = 1.0 * org_data['types'][:, :10].eq(0)  # Reverse mask
            useless_actions_x = torch.abs(_x) * mask_0
            useless_actions_y = torch.abs(_y) * mask_0
            useless_actions_loss = useless_actions_x + useless_actions_y
            useless_actions_loss = useless_actions_loss.sum(dim=1).mean()

            # Correct number of items - Trainable #
            noi = 1.0 * org_data['types'][:, :10].eq(1)
            noi = noi.cpu()
            noi = noi.long()
            noi = noi.sum(dim=1)
            noi = noi.cuda()
            how_many_loss = acc_loss(hmat, noi - 2)

            # total loss #
            total_loss = loss + useless_actions_loss + how_many_loss
            # total_loss = how_many_loss
            # Null Grad #

            total_loss.backward()
            if batch_idx % log_every == 0 and batch_idx > 0:
                _print(
                    f"Total Loss: {total_loss.detach().cpu().item()}")
                _print(
                    f"Total Loss: {total_loss.detach().cpu().item()} | Loss: {loss.detach().cpu().item()} |"
                    f" UAL : {useless_actions_loss.detach().cpu().item()} | Acc Loss: {how_many_loss.detach().cpu().item()}")
                _print(
                    f"Actions Max : {torch.max(mixed_actions)} | Actions Avg: {torch.mean(mixed_actions)} | Actions Min: {torch.min(mixed_actions)}")

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 50)
            optimizer.step()
            optimizer.zero_grad()
            batch_accuracy = 100 * (confusion_rewards.squeeze(1).mean()).item()
            accuracy_drop.append(batch_accuracy)
            epoch_accuracy_drop += batch_accuracy
            if batch_idx % log_every == 0 and batch_idx > 0:
                _print(
                    f"REINFORCE  {batch_idx} / {self.training_duration} | Accuracy Dropped By: {np.array(accuracy_drop)[-log_every:].mean()}%")
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

    def evaluate(self):
        self.model = self.model.to(self.device)
        self.model.zero_grad()
        self.model = self.model.eval()

        batch_idx = 0
        total_accuracy_drop = 0
        agent_choices = []

        while True:
            try:
                features, org_data, y_real = self.game.extract_features(self.dataloader_iter)
            except StopIteration:
                del self.dataloader_iter
                break
            mu, sigma_diag, hmat = self.model(features, org_data['question'])
            noi = 1.0 * org_data['types'][:, :10].eq(1)
            noi = noi.cpu()
            noi = noi.long()
            noi = noi.sum(dim=1)
            noi = noi.cuda()
            # cov_mat = torch.diag_embed(sigma_diag)
            # m = MultivariateNormal(loc=mu, covariance_matrix=cov_mat)
            # mixed_actions = m.sample()
            # Calc Reward #
            # rewards, confusion_rewards, _, altered_scene, _ = self.game.get_rewards(mixed_actions)
            # batch_accuracy = 100 * (confusion_rewards.squeeze(1).mean()).item()
            total_accuracy_drop += accuracy_metric(hmat, noi - 2)
            # agent_choices.append(mixed_actions)
            batch_idx += 1
            # if batch_accuracy > 0:
            #     bird_eye_view(batch_idx, x=kwarg_dict_to_device(org_data, 'cpu'), y=y_real.to('cpu').numpy(),
            #                   mode='before', q=0, a=0)
            #     bird_eye_view(batch_idx, x=altered_scene, y=None, mode='after', q=None, a=None)

        calc_acc_drop = total_accuracy_drop / batch_idx
        calc_acc_drop = round(calc_acc_drop, 3)
        _print(f"Total Accuracy Drop : {calc_acc_drop}")

        # plt.figure(figsize=(10, 10))
        # plt.title('REINFORCE Test Accuracy Drop Progress')
        # plt.plot(epoch_accuracy_drop_history, 'b')
        # plt.ylabel("Accuracy Drop on 96% State Transformer")
        # plt.xlabel("Epochs")
        # plt.savefig('./results/experiment_reinforce/epoch_progress.png')
        # plt.show()
        # plt.close()
