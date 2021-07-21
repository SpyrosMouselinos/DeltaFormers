import copy
import os.path as osp
import random
import sys

import numpy as np
import yaml

sys.path.insert(0, osp.abspath('.'))

import argparse
from torch.utils.data import Dataset
from modules.embedder import *
from utils.train_utils import StateCLEVR, ImageCLEVR, ImageCLEVR_HDF5
from bandit_modules.linucb import LinUCB
from bandit_modules.neuralucb import NeuralUCB


def _print(something):
    print(something, flush=True)
    return


AVAILABLE_DATASETS = {
    'DeltaRN': [StateCLEVR],
    'DeltaSQFormer': [StateCLEVR],
    'DeltaQFormer': [StateCLEVR],
    'DeltaSQFormerCross': [StateCLEVR],
    'DeltaSQFormerDisentangled': [StateCLEVR],
    'DeltaSQFormerLinear': [StateCLEVR],
    'DeltaRNFP': [ImageCLEVR, ImageCLEVR_HDF5],
}

AVAILABLE_MODELS = {'DeltaRN': DeltaRN,
                    'DeltaRNFP': DeltaRNFP,
                    'DeltaSQFormer': DeltaSQFormer,
                    'DeltaSQFormerCross': DeltaSQFormerCross,
                    'DeltaSQFormerDisentangled': DeltaSQFormerDisentangled,
                    'DeltaSQFormerLinear': DeltaSQFormerLinear,
                    'DeltaQFormer': DeltaQFormer}


def load(path: str, model: nn.Module):
    checkpoint = torch.load(path)
    # removes 'module' from dict entries, pytorch bug #3805
    if torch.cuda.device_count() >= 1 and any(k.startswith('module.') for k in checkpoint['model_state_dict'].keys()):
        checkpoint['model_state_dict'] = {k.replace('module.', ''): v for k, v in
                                          checkpoint['model_state_dict'].items()}
    model.load_state_dict(checkpoint['model_state_dict'])
    _print(f"Your model achieves {round(checkpoint['val_loss'], 4)} validation loss\n")
    return model


def kwarg_dict_to_device(data_obj, device):
    if device == 'cpu':
        return data_obj
    cpy = {}
    for key, _ in data_obj.items():
        cpy[key] = data_obj[key].to(device)
    return cpy


def accuracy_metric(y_pred, y_true):
    acc = (y_pred.argmax(1) == y_true).float().detach().cpu().numpy()
    return float(100 * acc.sum() / len(acc))


def get_fool_model(device, load_from=None, clvr_path='data/', questions_path='data/', scenes_path='data/',
                   use_cache=False, batch_size=128,
                   use_hdf5=False):
    if device == 'cuda':
        device = 'cuda:0'

    experiment_name = load_from.split('results/')[-1].split('/')[0]
    config = f'./results/{experiment_name}/config.yaml'
    with open(config, 'r') as fin:
        config = yaml.load(fin, Loader=yaml.FullLoader)

    model = AVAILABLE_MODELS[config['model_architecture']](config)
    _print(f"Loading Model of type: {config['model_architecture']}\n")
    model = model.to(device)
    model.eval()

    val_set = AVAILABLE_DATASETS[config['model_architecture']][0](config=config, split='val',
                                                                  clvr_path=clvr_path,
                                                                  questions_path=questions_path,
                                                                  scenes_path=scenes_path, use_cache=use_cache)

    val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                                 num_workers=0, shuffle=False, drop_last=True)

    model = load(path=load_from, model=model)

    return model, val_dataloader


def get_test_loader(load_from=None, clvr_path='data/', questions_path='data/', scenes_path='data/'):
    experiment_name = load_from.split('results/')[-1].split('/')[0]
    config = f'./results/{experiment_name}/config.yaml'
    with open(config, 'r') as fin:
        config = yaml.load(fin, Loader=yaml.FullLoader)

    test_set = AVAILABLE_DATASETS[config['model_architecture']][0](config=config, split='val',
                                                                   clvr_path=clvr_path,
                                                                   questions_path=questions_path,
                                                                   scenes_path=scenes_path, use_cache=False)

    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=1,
                                                  num_workers=0, shuffle=False, drop_last=True)
    return test_dataloader


class ContextualStatefulBandit:
    def __init__(self, testbed_model,
                 testbed_loader,
                 T=128,
                 n_arms=80,
                 confusion_model=None,
                 augmentation_strength=0.1,
                 seed=None,
                 device='cuda'
                 ):
        self._seed(seed)
        self.testbed_model = testbed_model
        self.testbed_model.eval()
        self.augmentation_strength = augmentation_strength
        # if confusion_model is None:
        #     self.confusion_model = self.testbed_model
        # else:
        #     self.confusion_model = confusion_model
        #     self.confusion_model.eval()
        self.testbed_loader = testbed_loader
        self.testbed_loader_iter = iter(testbed_loader)
        self.T = T
        self.n_arms = 80
        self.device = device

        # Generate features from testbed model's pre-ultimate layer
        self.reset()

    @property
    def arms(self):
        """Return [0, ...,n_arms-1]
        """
        return range(self.n_arms)

    def reset(self):
        """Generate new features and new rewards.
        """
        try:
            self.reset_features()
        except StopIteration:
            print("Dataset has ended... Reloading\n")
            self.testbed_loader_iter = iter(self.testbed_loader)
            self.reset_features()
        self.reset_rewards()

    def one_hot(self, a, num_classes):
        return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

    def create_arm_augmentation_vectors(self):
        # Petrurbation sets #
        # 8 perturbation options per object #
        pos_embeddings = self.one_hot(np.arange(0, 10), 10)[:, 1:]
        perturbation_options = np.array([
            [-1, -1, 0.0],
            [-1, 0.0, 0.0],
            [-1, 1, 0.0],
            [0, -1, 0.0],
            [0, 1, 0.0],
            [1, -1, 0.0],
            [1, 0, 0.0],
            [1, 1, 0.0]
        ]) * self.augmentation_strength
        po = perturbation_options.repeat(10, 0)
        po = np.concatenate([po, np.stack([pos_embeddings] * 8).reshape(80, 9)], 1)
        po = np.expand_dims(po, 0).repeat(self.T, 0)
        return po, perturbation_options

    def reset_T(self, bs):
        self.T = bs
        return

    def reset_features(self, external_loader=None):
        """Generate features from extraction model.
        """
        if external_loader is None:
            loader = self.testbed_loader_iter
        else:
            self.reset_T(1)
            loader = external_loader
        with torch.no_grad():
            if external_loader is None:
                val_batch = next(loader)
            else:
                val_batch = loader
            data, y_real = val_batch
            self.data = data
            self.answers = y_real
            data = kwarg_dict_to_device(data, self.device)
            y_preds, _, y_feats = self.testbed_model(**data)
            self.initial_predictions = y_preds.detach().cpu().argmax(1)
        self.features = y_feats.cpu().unsqueeze(1).numpy().repeat(self.n_arms, 1)
        self.po, self.perturbation_options = self.create_arm_augmentation_vectors()
        self.features = np.concatenate([self.features, self.po], axis=-1)
        self.features = self.features / np.linalg.norm(self.features, 2, 2, True)
        self.n_features = self.features.shape[-1]
        return self.features

    def alter_data(self):
        # Take self.data, alter them and forward pass again to get rewards #
        arm_predictions = []
        for object_index in range(0, 10):
            for perturbation_index in range(0, 8):
                self_data = copy.deepcopy(self.data)
                object_positions = copy.deepcopy(self.data['object_positions'])
                object_positions[:, object_index] += torch.FloatTensor(
                    np.expand_dims(self.perturbation_options[perturbation_index], 0).repeat(self.T, 0))

                self_data['object_positions'] = object_positions
                arm_predictions.append(
                    self.testbed_model(**kwarg_dict_to_device(self_data, self.device))[0].detach().cpu().argmax(1))
        return arm_predictions

    def reset_rewards(self):
        """Generate rewards for each arm and each round,
        following the reward function h + Gaussian noise.
        """
        self.rewards = np.zeros((self.T, self.n_arms))
        arm_predictions = self.alter_data()
        for i in range(0, self.n_arms):
            self.rewards[:, i] = 1 - ((self.answers.squeeze(1) - arm_predictions[i]).eq(0) * 1.0).numpy()
        return self.rewards

    def _seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)


def linUCBexperiment(args):
    model, loader = get_fool_model(device=args.device, load_from=args.load_from,
                                   scenes_path=args.scenes_path, questions_path=args.questions_path,
                                   clvr_path=args.clvr_path,
                                   use_cache=args.use_cache, use_hdf5=args.use_hdf5, batch_size=256)
    test_loader = get_test_loader(load_from=args.load_from,
                                  scenes_path=args.scenes_path, questions_path=args.questions_path,
                                  clvr_path=args.clvr_path)

    with open('./results_linucb.log', 'w+') as fout:
        ### Experiment 1 ###
        train_duration = 2500  # X Batch Size = 640_000
        test_duration = 5000  # X 1 = 5000
        cls = ContextualStatefulBandit(model, loader, 256, 80, None, augmentation_strength=1.0)
        gg = LinUCB(cls,
                    reg_factor=1.0,
                    delta=0.1,
                    confidence_scaling_factor=1.0,
                    )

        gg.run(epochs=train_duration)

        test_loader_iter = iter(test_loader)
        example_index = 0
        accuracy_drop = 0.0
        while example_index < test_duration:
            try:
                goo = next(test_loader_iter)
                test_features = cls.reset_features(goo)
                test_rewards = cls.reset_rewards()
                ucb, _, action = gg.test(test_features)
                accuracy_drop += test_rewards[0, action]
                example_index += 1
                if example_index % 500 == 0 and example_index > 0:
                    print(f"Scale 1 | Accuracy Dropped By: {100 * (accuracy_drop / example_index)}%")
            except StopIteration:
                break

        print(f"Scale 1 | Accuracy Dropped By: {100 * (accuracy_drop / test_duration)}%")
        fout.write(f"Scale 1 | Accuracy Dropped By: {100 * (accuracy_drop / test_duration)}%\n")

        ### Experiment 2 ###
        cls = ContextualStatefulBandit(model, loader, 256, 80, None, augmentation_strength=0.8)
        gg = LinUCB(cls,
                    reg_factor=1.0,
                    delta=0.1,
                    confidence_scaling_factor=1.0,
                    )

        gg.run(epochs=train_duration)
        test_loader_iter = iter(test_loader)
        example_index = 0
        accuracy_drop = 0.0
        while example_index < test_duration:
            try:
                goo = next(test_loader_iter)
                test_features = cls.reset_features(goo)
                test_rewards = cls.reset_rewards()
                ucb, _, action = gg.test(test_features)
                accuracy_drop += test_rewards[0, action]
                example_index += 1
                if example_index % 500 == 0 and example_index > 0:
                    print(f"Scale 0.8 | Accuracy Dropped By: {100 * (accuracy_drop / example_index)}%")
            except StopIteration:
                break
        print(f"Scale 0.8 | Accuracy Dropped By: {100 * (accuracy_drop / test_duration)}%")
        fout.write(f"Scale 0.8 | Accuracy Dropped By: {100 * (accuracy_drop / test_duration)}%\n")

        ### Experiment 3 ###
        cls = ContextualStatefulBandit(model, loader, 256, 80, None, augmentation_strength=0.5)
        gg = LinUCB(cls,
                    reg_factor=1.0,
                    delta=0.1,
                    confidence_scaling_factor=1.0,
                    )

        gg.run(epochs=train_duration)
        test_loader_iter = iter(test_loader)
        example_index = 0
        accuracy_drop = 0.0
        while example_index < test_duration:
            try:
                goo = next(test_loader_iter)
                test_features = cls.reset_features(goo)
                test_rewards = cls.reset_rewards()
                ucb, _, action = gg.test(test_features)
                accuracy_drop += test_rewards[0, action]
                example_index += 1
                if example_index % 500 == 0 and example_index > 0:
                    print(f"Scale 0.5 | Accuracy Dropped By: {100 * (accuracy_drop / example_index)}%")
            except StopIteration:
                break
        print(f"Scale 0.5 | Accuracy Dropped By: {100 * (accuracy_drop / test_duration)}%")
        fout.write(f"Scale 0.5 | Accuracy Dropped By: {100 * (accuracy_drop / test_duration)}%\n")

        ### Experiment 4 ###
        cls = ContextualStatefulBandit(model, loader, 256, 80, None, augmentation_strength=0.1)
        gg = LinUCB(cls,
                    reg_factor=1.0,
                    delta=0.1,
                    confidence_scaling_factor=1.0,
                    )

        gg.run(epochs=train_duration)

        test_loader_iter = iter(test_loader)
        example_index = 0
        accuracy_drop = 0.0
        while example_index < test_duration:
            try:
                goo = next(test_loader_iter)
                test_features = cls.reset_features(goo)
                test_rewards = cls.reset_rewards()
                ucb, _, action = gg.test(test_features)
                accuracy_drop += test_rewards[0, action]
                example_index += 1
                if example_index % 500 == 0 and example_index > 0:
                    print(f"Scale 0.1 | Accuracy Dropped By: {100 * (accuracy_drop / example_index)}%")
            except StopIteration:
                break
        print(f"Scale 0.1 | Accuracy Dropped By: {100 * (accuracy_drop / test_duration)}%")
        fout.write(f"Scale 0.1 | Accuracy Dropped By: {100 * (accuracy_drop / test_duration)}%\n")


def neuralUCBexperiment(args):
    BATCH_SIZE = 256
    model, loader = get_fool_model(device=args.device, load_from=args.load_from,
                                   scenes_path=args.scenes_path, questions_path=args.questions_path,
                                   clvr_path=args.clvr_path,
                                   use_cache=args.use_cache, use_hdf5=args.use_hdf5, batch_size=BATCH_SIZE)
    test_loader = get_test_loader(load_from=args.load_from,
                                  scenes_path=args.scenes_path, questions_path=args.questions_path,
                                  clvr_path=args.clvr_path)

    with open('./results_neuralucb.log', 'w+') as fout:
        ### Experiment 1 ###
        train_duration = 500  # X Batch Size = 128_000
        test_duration = 5000  # X 1 = 5000
        cls = ContextualStatefulBandit(model, loader, BATCH_SIZE, 80, None, augmentation_strength=1.0)
        gg = NeuralUCB(cls,
                       hidden_size=20,
                       reg_factor=1.0,
                       delta=0.1,
                       confidence_scaling_factor=1.0,
                       training_window=BATCH_SIZE // 2,
                       p=0.2,
                       learning_rate=0.01,
                       epochs=50,
                       train_every=(BATCH_SIZE // 4) - 1
                       )

        gg.run(epochs=train_duration)

        test_loader_iter = iter(test_loader)
        example_index = 0
        accuracy_drop = 0.0
        while example_index < test_duration:
            try:
                goo = next(test_loader_iter)
                test_features = cls.reset_features(goo)
                test_rewards = cls.reset_rewards()
                ucb, _, action = gg.test(test_features)
                accuracy_drop += test_rewards[0, action]
                example_index += 1
                if example_index % 500 == 0 and example_index > 0:
                    print(f"Scale 1 | Accuracy Dropped By: {100 * (accuracy_drop / example_index)}%")
            except StopIteration:
                break
        print(f"Scale 1 | Accuracy Dropped By: {100 * (accuracy_drop / test_duration)}%")
        fout.write(f"Scale 1 | Accuracy Dropped By: {100 * (accuracy_drop / test_duration)}%\n")

        ### Experiment 2 ###
        cls = ContextualStatefulBandit(model, loader, BATCH_SIZE, 80, None, augmentation_strength=0.5)
        gg = NeuralUCB(cls,
                       hidden_size=20,
                       reg_factor=1.0,
                       delta=0.1,
                       confidence_scaling_factor=1.0,
                       training_window=BATCH_SIZE // 2,
                       p=0.2,
                       learning_rate=0.01,
                       epochs=50,
                       train_every=(BATCH_SIZE // 4) - 1
                       )

        gg.run(epochs=train_duration)

        test_loader_iter = iter(test_loader)
        example_index = 0
        accuracy_drop = 0.0
        while example_index < test_duration:
            try:
                goo = next(test_loader_iter)
                test_features = cls.reset_features(goo)
                test_rewards = cls.reset_rewards()
                ucb, _, action = gg.test(test_features)
                accuracy_drop += test_rewards[0, action]
                example_index += 1
                if example_index % 500 == 0 and example_index > 0:
                    print(f"Scale 0.5 | Accuracy Dropped By: {100 * (accuracy_drop / example_index)}%")
            except StopIteration:
                break
        print(f"Scale 0.5 | Accuracy Dropped By: {100 * (accuracy_drop / test_duration)}%")
        fout.write(f"Scale 0.5 | Accuracy Dropped By: {100 * (accuracy_drop / test_duration)}%\n")

        ### Experiment 3 ###
        cls = ContextualStatefulBandit(model, loader, BATCH_SIZE, 80, None, augmentation_strength=0.1)
        gg = NeuralUCB(cls,
                       hidden_size=20,
                       reg_factor=1.0,
                       delta=0.1,
                       confidence_scaling_factor=1.0,
                       training_window=BATCH_SIZE // 2,
                       p=0.2,
                       learning_rate=0.01,
                       epochs=50,
                       train_every=(BATCH_SIZE // 4) - 1
                       )

        gg.run(epochs=train_duration)

        test_loader_iter = iter(test_loader)
        example_index = 0
        accuracy_drop = 0.0
        while example_index < test_duration:
            try:
                goo = next(test_loader_iter)
                test_features = cls.reset_features(goo)
                test_rewards = cls.reset_rewards()
                ucb, _, action = gg.test(test_features)
                accuracy_drop += test_rewards[0, action]
                example_index += 1
                if example_index % 500 == 0 and example_index > 0:
                    print(f"Scale 0.1 | Accuracy Dropped By: {100 * (accuracy_drop / example_index)}%")
            except StopIteration:
                break
        print(f"Scale 0.1 | Accuracy Dropped By: {100 * (accuracy_drop / test_duration)}%")
        fout.write(f"Scale 0.1 | Accuracy Dropped By: {100 * (accuracy_drop / test_duration)}%\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, help='cpu or cuda', default='cuda')
    parser.add_argument('--load_from', type=str, help='continue training',
                        default='./results/experiment_cross_sq/model.pt')
    parser.add_argument('--scenes_path', type=str, help='folder of scenes', default='data/')
    parser.add_argument('--questions_path', type=str, help='folder of questions', default='data/')
    parser.add_argument('--clvr_path', type=str, help='folder before images', default='data/')
    parser.add_argument('--use_cache', type=int, help='if to use cache (only in image clever)', default=0)
    parser.add_argument('--use_hdf5', type=int, help='if to use hdf5 loader', default=0)
    parser.add_argument('--mode', type=str, help='what kind of experiment to run', default='linear')
    args = parser.parse_args()

    if args.use_cache == 0:
        args.use_cache = False
    else:
        args.use_cache = True

    if args.use_hdf5 == 0:
        args.use_hdf5 = False
    else:
        args.use_hdf5 = True

    if args.mode == 'linear':
        linUCBexperiment(args)
    else:
        neuralUCBexperiment(args)
