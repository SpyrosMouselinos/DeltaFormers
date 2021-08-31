import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision


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


class ReplayMemory:
    def __init__(self, memory_size):
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
        self.fc1 = nn.Linear(input_size, input_size // 2)
        self.fc2 = nn.Linear(input_size // 2, input_size // 2)
        self.fc3 = nn.Linear(input_size // 2, input_size // 2)

        self.obj1 = nn.Linear(input_size // 2, input_size // 6)
        self.obj2 = nn.Linear(input_size // 2, input_size // 6)
        self.obj3 = nn.Linear(input_size // 2, input_size // 6)
        self.obj4 = nn.Linear(input_size // 2, input_size // 6)
        self.obj5 = nn.Linear(input_size // 2, input_size // 6)
        self.obj6 = nn.Linear(input_size // 2, input_size // 6)
        self.obj7 = nn.Linear(input_size // 2, input_size // 6)
        self.obj8 = nn.Linear(input_size // 2, input_size // 6)
        self.obj9 = nn.Linear(input_size // 2, input_size // 6)
        self.obj10 = nn.Linear(input_size // 2, input_size // 6)

        self.d1 = nn.Dropout(0.1)
        self.d2 = nn.Dropout(0.1)
        self.d3 = nn.Dropout(0.1)
        self.d4 = nn.Dropout(0.1)
        self.d5 = nn.Dropout(0.1)
        self.d6 = nn.Dropout(0.1)
        self.d7 = nn.Dropout(0.1)
        self.d8 = nn.Dropout(0.1)
        self.d9 = nn.Dropout(0.1)
        self.d10 = nn.Dropout(0.1)

        self.obj1x = nn.Linear(input_size // 6, 7)
        self.obj1y = nn.Linear(input_size // 6, 7)

        self.obj2x = nn.Linear(input_size // 6, 7)
        self.obj2y = nn.Linear(input_size // 6, 7)

        self.obj3x = nn.Linear(input_size // 6, 7)
        self.obj3y = nn.Linear(input_size // 6, 7)

        self.obj4x = nn.Linear(input_size // 6, 7)
        self.obj4y = nn.Linear(input_size // 6, 7)

        self.obj5x = nn.Linear(input_size // 6, 7)
        self.obj5y = nn.Linear(input_size // 6, 7)

        self.obj6x = nn.Linear(input_size // 6, 7)
        self.obj6y = nn.Linear(input_size // 6, 7)

        self.obj7x = nn.Linear(input_size // 6, 7)
        self.obj7y = nn.Linear(input_size // 6, 7)

        self.obj8x = nn.Linear(input_size // 6, 7)
        self.obj8y = nn.Linear(input_size // 6, 7)

        self.obj9x = nn.Linear(input_size // 6, 7)
        self.obj9y = nn.Linear(input_size // 6, 7)

        self.obj10x = nn.Linear(input_size // 6, 7)
        self.obj10y = nn.Linear(input_size // 6, 7)

    def forward(self, x):
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        x = F.relu(self.fc3(x), inplace=True)
        obj1 = F.relu(self.obj1(x), inplace=True)
        obj1 = self.d1(obj1)
        obj1x = self.obj1x(obj1)
        obj1y = self.obj1y(obj1)

        obj2 = F.relu(self.obj2(x), inplace=True)
        obj2 = self.d2(obj2)
        obj2x = self.obj2x(obj2)
        obj2y = self.obj2y(obj2)

        obj3 = F.relu(self.obj3(x), inplace=True)
        obj3 = self.d3(obj3)
        obj3x = self.obj3x(obj3)
        obj3y = self.obj3y(obj3)

        obj4 = F.relu(self.obj4(x), inplace=True)
        obj4 = self.d4(obj4)
        obj4x = self.obj4x(obj4)
        obj4y = self.obj4y(obj4)

        obj5 = F.relu(self.obj5(x), inplace=True)
        obj5 = self.d3(obj5)
        obj5x = self.obj5x(obj5)
        obj5y = self.obj5y(obj5)

        obj6 = F.relu(self.obj6(x), inplace=True)
        obj6 = self.d6(obj6)
        obj6x = self.obj6x(obj6)
        obj6y = self.obj6y(obj6)

        obj7 = F.relu(self.obj7(x), inplace=True)
        obj7 = self.d7(obj7)
        obj7x = self.obj7x(obj7)
        obj7y = self.obj7y(obj7)

        obj8 = F.relu(self.obj8(x), inplace=True)
        obj8 = self.d8(obj8)
        obj8x = self.obj8x(obj8)
        obj8y = self.obj8y(obj8)

        obj9 = F.relu(self.obj9(x), inplace=True)
        obj9 = self.d9(obj9)
        obj9x = self.obj9x(obj9)
        obj9y = self.obj9y(obj9)

        obj10 = F.relu(self.obj10(x), inplace=True)
        obj10 = self.d10(obj10)
        obj10x = self.obj1x(obj10)
        obj10y = self.obj1y(obj10)

        return (obj1x, obj2x, obj3x, obj4x, obj5x, obj6x, obj7x, obj8x, obj9x, obj10x), (
            obj1y, obj2y, obj3y, obj4y, obj5y, obj6y, obj7y, obj8y, obj9y, obj10y)

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
        # self.question_model = QuestionReintroduce(input_size, hidden_size, reverse_input)
        self.final_model = FFNet(512, dropout)

    def forward(self, x, q):
        # _, (q_summary, _) = self.question_model(q)
        # q_summary = torch.squeeze(q_summary, 0)
        # x = x / torch.norm(x, 2, 1, True)
        # q_summary = self.layer_norm_q(q_summary)
        # mix = torch.cat([x, q_summary], 1)
        mix = x
        sx, sy = self.final_model(mix)
        return sx, sy

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

    def quantize(self, action, effect_range=(-0.3, 0.3), steps=6):
        action = action.detach().cpu().numpy()
        bs_, length_ = action.shape
        quantized_actions = np.empty(action.shape)
        for i in range(bs_):
            for j in range(length_):
                quantized_actions[i, j] = effect_range[0] + action[i, j] * ((effect_range[1] - effect_range[0]) / steps)
        return quantized_actions

    def train(self, log_every=100, save_every=10_000):
        t = 100
        self.model = self.model.to(self.device)
        self.model = self.model.train()
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

        optimizer.zero_grad()
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

            sx, sy = self.model(
                features, org_data['question'])
            action_probs_x = torch.distributions.Categorical(
                torch.softmax(torch.cat([f.unsqueeze(1) for f in sx], dim=1) / t, dim=2))
            action_probs_y = torch.distributions.Categorical(
                torch.softmax(torch.cat([f.unsqueeze(1) for f in sy], dim=1)/  t, dim=2))

            actionsx = action_probs_x.sample()
            actionsy = action_probs_y.sample()

            log_probs = action_probs_x.log_prob(actionsx) + action_probs_y.log_prob(actionsy)
            log_probs = log_probs.sum(1)
            action = torch.cat([actionsx, actionsy], dim=1)

            mixed_actions = self.quantize(action)
            rewards_, confusion_rewards, change_rewards, fail_rewards, invalid_scene_rewards, scene, predictions_after = self.game.get_rewards(
                mixed_actions)
            rewards = torch.FloatTensor(rewards_).squeeze(1).to(self.device)
            #rm = rewards.mean()
            # if rewards.std().item() > 1:
            #     rs = rewards.std()
            # else:
            #     rs = 1
            # rewards = rewards  / rs
            loss = -log_probs * rewards
            loss = loss.mean()
            total_loss = loss

            total_loss.backward()
            if batch_idx % log_every == 0 and batch_idx > 0:
                _print(
                    f"Total Loss: {total_loss.detach().cpu().item()} | Total Reward: {rewards_.sum()}")

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
