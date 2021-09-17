import itertools
import json
import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

with open(f'{osp.abspath(".")}/data/vocab.json', 'r') as fin:
    parsed_json = json.load(fin)
    q2index = parsed_json['question_token_to_idx']
    a2index = parsed_json['answer_token_to_idx']

index2q = {v: k for k, v in q2index.items()}
index2a = {v: k for k, v in a2index.items()}


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
        # self.fc2 = nn.Linear(input_size // 2, input_size // 2)
        # self.fc3 = nn.Linear(input_size // 2, input_size // 2)

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
        # x = F.relu(self.fc2(x), inplace=True)
        # x = F.relu(self.fc3(x), inplace=True)
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


class BVNet(nn.Module):
    """
    Baseline REINFORCE - Value Calculating Network
    """

    def __init__(self, input_size):
        super(BVNet, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, input_size // 2)
        self.dr1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(input_size // 2, input_size // 4)
        self.dr2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(input_size // 4, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x), inplace=True)
        x = self.dr1(x)
        x = F.relu(self.fc2(x), inplace=True)
        x = self.dr2(x)
        x = torch.tanh(self.fc3(x)) * 1.5
        return x


class PolicyNet(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.0, reverse_input=False):
        super(PolicyNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reverse_input = reverse_input
        self.dropout = dropout
        # self.question_model = QuestionReintroduce(input_size, hidden_size, reverse_input)
        self.final_model = FFNet(512, dropout)
        self.value_model = BVNet(512)

    def forward(self, x, q):
        # _, (q_summary, _) = self.question_model(q)
        # q_summary = torch.squeeze(q_summary, 0)
        # x = x / torch.norm(x, 2, 1, True)
        # q_summary = self.layer_norm_q(q_summary)
        # mix = torch.cat([x, q_summary], 1)
        mix = x
        sx, sy = self.final_model(mix)
        state_value = self.value_model(mix)
        return sx, sy, state_value

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


class Wizard:
    def __init__(self, n_objects, n_dof=7, n_actions=2):
        self.n_objects = n_objects
        self.n_dof = n_dof
        single_object_template = list(range(0, n_dof ** n_actions))
        multi_object_template = tuple([single_object_template] * self.n_objects)
        self.action_memory = list(itertools.product(*multi_object_template))
        self.registered_actions = []

    def restart(self, n_objects, n_dof=7, n_actions=2):
        self.n_objects = n_objects
        self.n_dof = n_dof
        single_object_template = list(range(0, n_dof ** n_actions))
        multi_object_template = tuple([single_object_template] * self.n_objects)
        self.action_memory = list(itertools.product(*multi_object_template))

    def act(self, action_id):
        actions = self.action_memory[action_id]
        actions_x = torch.LongTensor([f % self.n_dof for f in actions])
        actions_y = torch.LongTensor([f // self.n_dof for f in actions])
        return actions_x, actions_y

    def register(self, example_id, action_id):
        self.registered_actions.append((example_id, action_id))
        return


class Re1nforceTrainer:
    def __init__(self, model, game, dataloader, device='cuda', lr=0.001, train_duration=10, batch_size=32,
                 batch_tolerance=1, batch_reset=100, name='', predictions_before_pre_calc=None, resnet=None,
                 fool_model_name=None):
        self.name = name
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
        self.predictions_before_pre_calc = predictions_before_pre_calc
        self.resnet = resnet
        self.fool_model_name = fool_model_name

    def quantize(self, action, effect_range=(-0.3, 0.3), steps=6):
        action = action.detach().cpu().numpy()
        bs_, length_ = action.shape
        quantized_actions = np.empty(action.shape)
        for i in range(bs_):
            for j in range(length_):
                quantized_actions[i, j] = effect_range[0] + action[i, j] * ((effect_range[1] - effect_range[0]) / steps)
        return quantized_actions

    def train(self, log_every=100, save_every=10_000, logger=None):
        if self.fool_model_name not in ['sa', 'iep', 'film', 'rnfp']:
            prefix = 'state'
        else:
            prefix = 'visual'
        t = 10
        self.model = self.model.to(self.device)
        self.model = self.model.train()
        if self.lr >= 5e-4:
            optimizer = optim.AdamW(
                [{'params': self.model.final_model.parameters(), 'lr': self.lr * 0.9, 'weight_decay': 1e-4},
                 {'params': self.model.value_model.parameters(), 'lr': self.lr * 0.1, 'weight_decay': 1e-4},
                 ])
        else:
            optimizer = optim.AdamW(
                [{'params': self.model.final_model.parameters(), 'lr': self.lr, 'weight_decay': 1e-4},
                 {'params': self.model.value_model.parameters(), 'lr': self.lr, 'weight_decay': 1e-4},
                 ])

        limit = 1
        accuracy_drop = []
        confusion_drop = []
        batch_idx = 0
        epochs_passed = 0
        epoch_accuracy_drop = 0
        epoch_confusion_drop = 0
        patience = 30
        epoch_accuracy_drop_history = []
        epoch_confusion_drop_history = []

        while epochs_passed < self.training_duration:
            try:
                features, org_data, _ = self.game.extract_features(self.dataloader_iter)
                if self.predictions_before_pre_calc is not None:
                    self.current_predictions_before = self.predictions_before_pre_calc[
                                                      (batch_idx % len(self.dataloader)) * self.batch_size:((
                                                                                                                    batch_idx % len(
                                                                                                                self.dataloader)) + 1) * self.batch_size]
                else:
                    self.current_predictions_before = None
            except StopIteration:
                del self.dataloader_iter
                self.dataloader_iter = iter(self.dataloader)
                features, org_data, _ = self.game.extract_features(self.dataloader_iter)
                if self.predictions_before_pre_calc is not None:
                    self.current_predictions_before = self.predictions_before_pre_calc[
                                                      (batch_idx % len(self.dataloader)) * self.batch_size:((
                                                                                                                    batch_idx % len(
                                                                                                                self.dataloader)) + 1) * self.batch_size]
                else:
                    self.current_predictions_before = None
                best_epoch_accuracy_drop = epoch_accuracy_drop / len(self.dataloader)
                best_epoch_confusion_drop = epoch_confusion_drop / len(self.dataloader)
                if len(epoch_accuracy_drop_history) > 0 and best_epoch_accuracy_drop <= sum(
                        epoch_accuracy_drop_history[-15:]) / len(epoch_accuracy_drop_history[-15:]):
                    patience -= 1

                if best_epoch_confusion_drop > limit:
                    self.model.save(
                        f'./results/experiment_reinforce/{prefix}/model_reinforce_{self.name}_{self.fool_model_name}_{round(best_epoch_confusion_drop, 1)}.pt')
                    logger.session['rl_agent'].upload(
                        f'./results/experiment_reinforce/{prefix}/model_reinforce_{self.name}_{self.fool_model_name}_{round(best_epoch_confusion_drop, 1)}.pt')
                    limit += 1
                _print(
                    f"REINFORCE 2  Epoch {epochs_passed} | Epoch Accuracy Drop: {best_epoch_accuracy_drop}% | Epoch Confusion {best_epoch_confusion_drop} % | Patience: {patience}")
                if logger is not None:
                    logger.log({'Epoch': epochs_passed, 'Drop': best_epoch_accuracy_drop,
                                'Consistency': best_epoch_confusion_drop})
                epochs_passed += 1
                epoch_accuracy_drop_history.append(epoch_accuracy_drop / len(self.dataloader))
                epoch_confusion_drop_history.append(epoch_confusion_drop / len(self.dataloader))
                epoch_accuracy_drop = 0
                epoch_confusion_drop = 0

            # Forward Pass #
            try:
                sx, sy, state_values = self.model(
                    features, org_data['question'])
                action_probs_x = torch.distributions.Categorical(
                    torch.softmax(torch.cat([f.unsqueeze(1) for f in sx], dim=1) / t, dim=2))
                action_probs_y = torch.distributions.Categorical(
                    torch.softmax(torch.cat([f.unsqueeze(1) for f in sy], dim=1) / t, dim=2))

                actionsx = action_probs_x.sample()
                actionsy = action_probs_y.sample()

                log_probs = action_probs_x.log_prob(actionsx) + action_probs_y.log_prob(actionsy)
                log_probs = log_probs.sum(1)
                action = torch.cat([actionsx, actionsy], dim=1)

                mixed_actions = self.quantize(action)
                rewards_, confusion_rewards, change_rewards, fail_rewards, invalid_scene_rewards, scene, predictions_after, state_after = self.game.get_rewards(
                    action_vector=mixed_actions, current_predictions_before=self.current_predictions_before,
                    resnet=self.resnet)
                rewards = torch.FloatTensor(rewards_).to(self.device)
                advantages = rewards - state_values.squeeze(1).detach()

                ploss = -log_probs * advantages
                vloss = (state_values.squeeze(1) - rewards.detach()) ** 2
                loss = ploss + vloss
                loss = loss.mean()
                total_loss = loss

                total_loss.backward()
                if batch_idx % log_every == 0 and batch_idx > 0 and log_every != -1:
                    _print(
                        f"Total Loss: {total_loss.detach().cpu().item()} | Total Reward: {rewards_.sum()}")

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 50)
                optimizer.step()
                optimizer.zero_grad()
                batch_accuracy = 100 * (confusion_rewards.mean()).item()
                batch_confusion = 100 * (change_rewards.mean()).item()
                accuracy_drop.append(batch_accuracy)
                confusion_drop.append(batch_confusion)
                epoch_accuracy_drop += batch_accuracy
                epoch_confusion_drop += batch_confusion
                if epochs_passed % log_every == 0 and epochs_passed > 0 and log_every != -1:
                    _print(
                        f"REINFORCE2 {batch_idx} / {self.training_duration} | Accuracy Dropped By: {np.array(accuracy_drop)[-log_every:].mean()}% | Model confused: {np.array(confusion_drop)[-log_every:].mean()}")
                if epochs_passed % save_every == 0 and epochs_passed > 0:
                    self.model.save(
                        f'./results/experiment_reinforce/{prefix}/model_reinforce_{self.name}_{self.fool_model_name}.pt')
                if patience == 0:
                    break
                batch_idx += 1
            except KeyboardInterrupt:
                _print("Handling Graceful Exit...\n")
                break

        self.model.save(
            f'./results/experiment_reinforce/{prefix}/model_reinforce_{self.name}_{self.fool_model_name}.pt')

        # fig1 = plt.figure(figsize=(10, 10))
        # plt.title(f'REINFORCE Epoch Consistency Drop on {self.fool_model_name}')
        # plt.plot(epoch_confusion_drop_history, 'b')
        # plt.ylabel("Consistency Drop")
        # plt.xlabel("Epochs")
        # plt.savefig(f'./results/experiment_reinforce/epoch_consistency{self.name}_{self.fool_model_name}.png')
        # if logger is not None:
        #     logger.image_store('consistency', fig1)
        # plt.show()
        # plt.close()
        #
        # plt.figure(figsize=(10, 10))
        # plt.title(f'REINFORCE Epoch Accuracy Drop on {self.fool_model_name}')
        # plt.plot(epoch_accuracy_drop_history, 'b')
        # plt.ylabel("Accuracy Drop")
        # plt.xlabel("Epochs")
        # plt.savefig(f'./results/experiment_reinforce/epoch_drop{self.name}_{self.fool_model_name}.png')
        # if logger is not None:
        #     logger.image_store('drop', fig1)
        # plt.show()
        # plt.close()
        if logger is not None:
            logger.log({'Epoch': epochs_passed, 'Drop': max(epoch_accuracy_drop_history),
                        'Consistency': max(epoch_confusion_drop_history)})
        return max(epoch_accuracy_drop_history), max(epoch_confusion_drop_history)

    def print_over(self, index, path, question, oa, aa):
        ba = f"Before: {oa}"
        oa = f"GT: {oa}"
        if aa is not None:
            aa = f"After: {aa}"

        def split_q(q):
            if len(q) > 30:
                q = q[:30] + '\n' + split_q(q[30:])
            return q

        img = plt.imread(path)
        plt.figure(figsize=(8, 8))
        plt.grid(None)
        plt.imshow(img)
        plt.text(0, 30, split_q(question), bbox=dict(fill=False, edgecolor='red', linewidth=1))
        plt.text(85, 15, split_q(oa), bbox=dict(fill=False, edgecolor='green', linewidth=1))
        plt.text(85, 35, split_q(ba), bbox=dict(fill=False, edgecolor='blue', linewidth=1))
        if aa is not None:
            plt.text(85, 55, split_q(aa), bbox=dict(fill=False, edgecolor='yellow', linewidth=1))
        plt.savefig(f'./results/images/{index}_{self.fool_model_name}.png')
        plt.close()
        return

    def evaluate(self, example_range=(0, 1280)):
        self.model = self.model.to(self.device)
        self.model.zero_grad()
        self.model = self.model.eval()

        batch_idx = 0
        n_burn_iterations = example_range[0] // self.batch_size
        n_eligible_iterations = (example_range[1] - example_range[0]) // self.batch_size
        while batch_idx < n_burn_iterations:
            _ = next(self.dataloader_iter)
            batch_idx += 1

        batch_idx = 0
        drop = 0
        confusion = 0
        worth_discovering = 0
        while batch_idx < n_eligible_iterations:
            try:
                features, org_data, y_real = self.game.extract_features(self.dataloader_iter)
                if self.predictions_before_pre_calc is not None:
                    self.current_predictions_before = self.predictions_before_pre_calc[
                                                      (batch_idx % len(self.dataloader)) * self.batch_size:((
                                                                                                                    batch_idx % len(
                                                                                                                self.dataloader)) + 1) * self.batch_size]
                else:
                    self.current_predictions_before = None
            except StopIteration:
                del self.dataloader_iter
                self.dataloader_iter = iter(self.dataloader)
                break
            sx, sy, state_values = self.model(
                features, org_data['question'])
            actionsx = torch.cat([f.unsqueeze(1) for f in sx], dim=1).max(2)[1]
            actionsy = torch.cat([f.unsqueeze(1) for f in sy], dim=1).max(2)[1]

            action = torch.cat([actionsx, actionsy], dim=1)
            # Calc Reward #
            mixed_actions = self.quantize(action)
            rewards_, confusion_rewards, change_rewards, fail_rewards, invalid_scene_rewards, scene, predictions_after, state_after = self.game.get_rewards(
                action_vector=mixed_actions, current_predictions_before=self.current_predictions_before,
                resnet=self.resnet)
            state_values = state_values.squeeze(1).detach().cpu().numpy()
            drop += (confusion_rewards > 0).sum()
            confusion += (change_rewards > 0).sum()
            worth_discovering += (state_values + 0.1 > 0.05).sum()

            if (confusion_rewards > 0).sum() > 0:
                # final_images, final_questions = self.game.state2img(kwarg_dict_to_device(org_data, 'cpu'), delete_every=False, bypass=False,
                #                    retry=True, custom_index=batch_idx)
                if batch_idx < 10:
                    name = '0' + str(batch_idx)
                else:
                    name = '10'
                # if len(final_images) == 1:
                #    self.print_over(f'C:\\Users\\Guldan\\Desktop\\DeltaFormers\\neural_render\\images\\CLEVR_Rendered_0000{name}.png', ' '.join([index2q[f] for f in final_questions[0].numpy() if f != 0][1:-1]), index2a[y_real.to('cpu').numpy()[0,0] + 4], None)
                final_images, final_questions = self.game.state2img(kwarg_dict_to_device(state_after, 'cpu'),
                                                                    delete_every=False, bypass=True,
                                                                    custom_index=batch_idx)
                if len(final_images) == 1:
                    self.print_over(batch_idx, f'./neural_render/images/CLEVR_Rendered_0000{name}.png',
                                    ' '.join([index2q[f] for f in final_questions[0].numpy() if f != 0][1:-1]),
                                    index2a[y_real.to('cpu').numpy()[0, 0] + 4],
                                    index2a[predictions_after.to('cpu').numpy()[0] + 4])
                # q = bird_eye_view(batch_idx, x=kwarg_dict_to_device(org_data, 'cpu'), y=y_real.to('cpu').numpy(),
                #                    mode='before', q=0, a=0)
                # _ = bird_eye_view(batch_idx, x=scene, y=y_real.to('cpu').numpy(), mode='after', q=q,
                #                   a=predictions_after)

            batch_idx += 1

        calc_acc_drop = 100 * drop / (example_range[1] - example_range[0])
        calc_acc_drop = round(calc_acc_drop.item(), 3)
        calc_acc_confusion = 100 * confusion / (example_range[1] - example_range[0])
        calc_acc_confusion = round(calc_acc_confusion.item(), 3)
        worth_disc_n = worth_discovering
        _print(f"Total Accuracy Drop : {calc_acc_drop}")
        _print(f"Total Accuracy Confusion : {calc_acc_confusion}")
        _print(f"Total Items Worth Discovering  : {worth_disc_n} out of {(example_range[1] - example_range[0])}")

    def discover(self, log_every=100, save_every=10_000, example_range=(0, 1000)):
        accuracy_drop = []
        confusion_drop = []
        batch_idx = 0
        epochs_passed = 0
        epoch_accuracy_drop = 0
        epoch_confusion_drop = 0
        epoch_accuracy_drop_history = []
        epoch_confusion_drop_history = []
        wizard = Wizard(1)
        while batch_idx < self.training_duration:
            try:
                features, org_data, _ = self.game.extract_features(self.dataloader_iter)
                wizard.restart(org_data['types'][:, :10].sum(1))
            except StopIteration:
                del self.dataloader_iter
                self.dataloader_iter = iter(self.dataloader)
                features, org_data, _ = self.game.extract_features(self.dataloader_iter)
                wizard.restart(org_data['types'][:, :10].sum(1))
                best_epoch_accuracy_drop = epoch_accuracy_drop / len(self.dataloader)
                best_epoch_confusion_drop = epoch_confusion_drop / len(self.dataloader)
                _print(
                    f"REINFORCE 2  Epoch {epochs_passed} | Epoch Accuracy Drop: {best_epoch_accuracy_drop}% | Epoch Confusion {best_epoch_confusion_drop} %")
                epochs_passed += 1
                epoch_accuracy_drop_history.append(epoch_accuracy_drop / len(self.dataloader))
                epoch_confusion_drop_history.append(epoch_confusion_drop / len(self.dataloader))
                epoch_accuracy_drop = 0
                epoch_confusion_drop = 0

            for i in range(len(wizard.action_memory)):
                actionsx, actionsy = wizard.act(i)
                action = torch.cat([actionsx, actionsy], dim=0).unsqueeze(1)
                mixed_actions = self.quantize(action)
                rewards_, confusion_rewards, change_rewards, _, _, _, _ = self.game.get_rewards(
                    mixed_actions)

                if rewards_ == 1:
                    wizard.register(batch_idx, i)
                    batch_accuracy = 100 * (confusion_rewards.squeeze(1).mean()).item()
                    batch_confusion = 100 * (change_rewards.squeeze(1).mean()).item()
                    accuracy_drop.append(batch_accuracy)
                    confusion_drop.append(batch_confusion)
                    epoch_accuracy_drop += batch_accuracy
                    epoch_confusion_drop += batch_confusion
                    break

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
        plt.savefig(f'./results/experiment_reinforce/epoch_progress{self.name}.png')
        plt.show()
        plt.close()
