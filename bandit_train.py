import ast
import copy
import json
import os
import os.path as osp
import pickle
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
from oracle.Oracle_CLEVR import Oracle


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

with open('./translation_tables.yaml', 'r') as fin:
    translation_tables = yaml.load(fin, Loader=yaml.FullLoader)

with open('./data/vocab.json', 'r') as fin:
    vocabs = json.load(fin)
    back_translation = vocabs['answer_token_to_idx']
    back_translation.update({'true': back_translation['yes']})
    back_translation.update({'false': back_translation['no']})
    back_translation.update({'__invalid__': 3})  # So after -4 goes to -1


def translate_state(example_state):
    grouped_templates = []

    def header_template():
        return dict({'split': 'Val', 'directions': {'below': [-0.0, -0.0, -1.0],
                                                    'front': [0.754490315914154, -0.6563112735748291, -0.0],
                                                    'above': [0.0, 0.0, 1.0],
                                                    'right': [0.6563112735748291, 0.7544902563095093, -0.0],
                                                    'behind': [-0.754490315914154, 0.6563112735748291, 0.0],
                                                    'left': [-0.6563112735748291, -0.7544902563095093, 0.0]},
                     "objects": []}
                    )

    def object_template(position, color, shape, material, size):
        return {
            "3d_coords": list(position),
            "shape": translation_tables['reverse_translation_shape'][shape],
            "size": translation_tables['reverse_translation_size'][size],
            "color": translation_tables['reverse_translation_color'][color],
            "material": translation_tables['reverse_translation_material'][material]
        }

    batch_size = example_state['types'].numpy().shape[0]

    for i in range(batch_size):
        template = header_template()
        number_of_objects = np.where(example_state['types'][i].numpy() == 1)[0]
        object_positions = example_state['object_positions'][i].numpy()[number_of_objects]
        object_colors = example_state['object_colors'][i].numpy()[number_of_objects]
        object_shapes = example_state['object_shapes'][i].numpy()[number_of_objects]
        object_materials = example_state['object_materials'][i].numpy()[number_of_objects]
        object_sizes = example_state['object_sizes'][i].numpy()[number_of_objects]
        for p, c, s, m, z in zip(object_positions, object_colors, object_shapes, object_materials, object_sizes):
            template['objects'].append(object_template(p, c, s, m, z))
        grouped_templates.append(template)
    if len(grouped_templates) == 1:
        return grouped_templates[0]
    return grouped_templates


def translate_program(example_program):
    program = []
    for instruction in example_program:
        new_instruction = {}
        for entry, value in instruction.items():
            if len(value) > 0:
                if isinstance(value[0], torch.Tensor):
                    value = [f.cpu().item() for f in value]
                elif isinstance(value[0], tuple):
                    value = [value[0][0]]
                else:
                    value = value[0]
            new_instruction.update({entry: value})
        program.append(new_instruction)
    return program


def translate_str_program(example_program):
    final_programs = []
    program_str = []
    for example in example_program:
        for char in example:
            if char == 'P':
                continue
            else:
                program_str.append(char)
        program = ast.literal_eval(''.join(program_str))
        program_str = []
        final_programs.append(program)
    if len(final_programs) == 1:
        final_programs = final_programs[0]
    return final_programs


def translate_answer(answer):
    answers = []
    if isinstance(answer, list):
        for answer_ in answer:
            answer_idx = back_translation[answer_] - 4
            answers.append(torch.LongTensor([answer_idx]))
    else:
        answer_idx = back_translation[answer] - 4
        answers.append(torch.LongTensor([answer_idx]))
    if len(answers) == 1:
        return answers[0]
    return answers


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
    model = load(path=load_from, model=model)
    model = model.to(device)
    model.eval()

    config_fool = f'./results/experiment_linear_sq/config.yaml'
    with open(config_fool, 'r') as fin:
        config_fool = yaml.load(fin, Loader=yaml.FullLoader)

    model_fool = AVAILABLE_MODELS[config_fool['model_architecture']](config_fool)
    _print(f"Loading Model of type: {config_fool['model_architecture']}\n")
    model_fool = load(path=f'./results/experiment_linear_sq/model.pt', model=model_fool)
    model_fool = model_fool.to(device)
    model_fool.eval()

    val_set = AVAILABLE_DATASETS[config['model_architecture']][0](config=config, split='val',
                                                                  clvr_path=clvr_path,
                                                                  questions_path=questions_path,
                                                                  scenes_path=scenes_path, use_cache=use_cache,
                                                                  return_program=True)

    val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                                 num_workers=0, shuffle=True, drop_last=True)

    _print(f"Loader has : {len(val_dataloader)} batches\n")
    return model, model_fool, val_dataloader


def get_test_loader(load_from=None, clvr_path='data/', questions_path='data/', scenes_path='data/'):
    experiment_name = load_from.split('results/')[-1].split('/')[0]
    config = f'./results/{experiment_name}/config.yaml'
    with open(config, 'r') as fin:
        config = yaml.load(fin, Loader=yaml.FullLoader)

    test_set = AVAILABLE_DATASETS[config['model_architecture']][0](config=config, split='val',
                                                                   clvr_path=clvr_path,
                                                                   questions_path=questions_path,
                                                                   scenes_path=scenes_path, use_cache=False,
                                                                   return_program=True)

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
        self.compensation_for_state_scale = 3
        if confusion_model is None:
            self.confusion_model = self.testbed_model
        else:
            self.confusion_model = confusion_model
            self.confusion_model.eval()
        self.testbed_loader = testbed_loader
        self.testbed_loader_iter = iter(testbed_loader)
        self.T = T
        self.n_arms = 80
        self.device = device
        self.oracle = Oracle(metadata_path='./metadata.json')
        self.internal_dataset = {'X_scenes': [], 'Y_rewards': [], 'Y_values': []}
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
        ]) * self.augmentation_strength / self.compensation_for_state_scale
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
            data, y_real, programs = val_batch
            self.data = data
            self.answers = y_real
            self.programs = programs
            data = kwarg_dict_to_device(data, self.device)
            y_preds, _, y_feats = self.testbed_model(**data)
            self.initial_predictions = y_preds.detach().cpu().argmax(1)

        self.internal_dataset['X_scenes'].append(y_feats.cpu().numpy())
        self.features = y_feats.cpu().unsqueeze(1).numpy().repeat(self.n_arms, 1)
        self.po, self.perturbation_options = self.create_arm_augmentation_vectors()
        self.features = np.concatenate([self.features, self.po], axis=-1)
        self.features = self.features / np.linalg.norm(self.features, 2, 2, True)
        self.n_features = self.features.shape[-1]
        return self.features

    def alter_data(self, specific_action=None):
        # Take self.data, alter them and forward pass again to get rewards #
        if specific_action is None:
            arm_predictions_after = []
            arm_validity = []
            arm_predictions_before = self.confusion_model(**kwarg_dict_to_device(self.data, self.device))[
                0].detach().cpu().argmax(1)

            programs = translate_str_program(self.programs)
            for object_index in range(0, 10):
                for perturbation_index in range(0, 8):
                    self_data = copy.deepcopy(self.data)
                    object_positions = copy.deepcopy(self.data['object_positions'])
                    object_positions[:, object_index] += torch.FloatTensor(
                        np.expand_dims(self.perturbation_options[perturbation_index], 0).repeat(self.T, 0))

                    self_data['object_positions'] = object_positions
                    scene = translate_state(self_data)
                    res = self.oracle(programs, scene, None)
                    res = translate_answer(res)
                    arm_predictions_after.append(
                        self.confusion_model(**kwarg_dict_to_device(self_data, self.device))[0].detach().cpu().argmax(
                            1))
                    arm_validity.append(res)

            return arm_predictions_after, arm_predictions_before, arm_validity
        else:
            arm_predictions_after = []
            arm_validity = []
            arm_predictions_before = self.confusion_model(**kwarg_dict_to_device(self.data, self.device))[
                0].detach().cpu().argmax(1)
            programs = translate_str_program(self.programs)
            object_index, perturbation_index = specific_action // 8, specific_action % 8
            self_data = copy.deepcopy(self.data)
            object_positions = copy.deepcopy(self.data['object_positions'])
            object_positions[:, object_index] += torch.FloatTensor(
                np.expand_dims(self.perturbation_options[perturbation_index], 0).repeat(self.T, 0))

            self_data['object_positions'] = object_positions
            scene = translate_state(self_data)
            res = self.oracle(programs, scene, None)
            res = translate_answer(res)
            arm_predictions_after.append(
                self.confusion_model(**kwarg_dict_to_device(self_data, self.device))[0].detach().cpu().argmax(
                    1))
            arm_validity.append(res)

            return arm_predictions_after, arm_predictions_before, arm_validity

    def reset_rewards(self, specific_action=None):
        """Generate rewards for each arm and each round,
        following the reward function h + Gaussian noise.
        """
        if specific_action is None:
            self.rewards = np.zeros((self.T, self.n_arms))
            arm_predictions_after, arm_predictions_before, arm_validity = self.alter_data(specific_action=None)
            arm_predictions_before = arm_predictions_before.unsqueeze(1)
            arm_predictions_after = torch.stack(arm_predictions_after, dim=1).long()
            if self.T > 1:
                arm_validity = [torch.LongTensor(f) for f in arm_validity]
            answer_stayed_the_same = self.answers - torch.stack(arm_validity, dim=1).long()
            answer_stayed_the_same = 1.0 * answer_stayed_the_same.eq(0)
            model_answered_correctly = self.answers - arm_predictions_before
            model_answered_correctly = 1.0 * model_answered_correctly.eq(0)
            # Loss Rule #
            # 0) If we had a correct answer in the first place and is now false -> Reward 1
            # 1) If we had a correct answer in the first place and it is still correct -> Reward 0
            # 2) If we had a false answer in the first place -> Reward 0
            # 3) If the state is invalid - Reward - 1
            # for i in range(0, self.n_arms):
            #     self.rewards[:, i] = 1 - ((self.answers.squeeze(1) - arm_predictions[i]).eq(0) * 1.0).numpy()
            self.rewards = (model_answered_correctly * answer_stayed_the_same * (
                    1.0 - 1.0 * (arm_predictions_before - arm_predictions_after).eq(0))).numpy()
            # + ((1 - answer_stayed_the_same) * np.ones_like(self.rewards) * -1.0).numpy()

            self.internal_dataset['Y_rewards'].append(np.argmax(self.rewards, 1))
            self.internal_dataset['Y_values'].append(np.max(self.rewards, 1))
            if len(self.internal_dataset['Y_values']) >= 20:
                path_id = 0
                path = f'./results/data_lin_part_{path_id}.pt'
                while os.path.exists(path):
                    path_id += 1
                    path = f'./results/data_lin_part_{path_id}.pt'

                with open(path, 'wb') as f:
                    pickle.dump(self.internal_dataset, f)
                del self.internal_dataset
                self.internal_dataset = {'X_scenes': [], 'Y_rewards': [], 'Y_values': []}
            return self.rewards
        else:
            self.rewards = np.zeros((self.T, 1))
            arm_predictions_after, arm_predictions_before, arm_validity = self.alter_data(
                specific_action=specific_action)
            arm_predictions_before = arm_predictions_before.unsqueeze(1)
            arm_predictions_after = torch.stack(arm_predictions_after, dim=1).long()
            if self.T > 1:
                arm_validity = [torch.LongTensor(f) for f in arm_validity]
            answer_stayed_the_same = self.answers - torch.stack(arm_validity, dim=1).long()
            answer_stayed_the_same = 1.0 * answer_stayed_the_same.eq(0)
            model_answered_correctly = self.answers - arm_predictions_before
            model_answered_correctly = 1.0 * model_answered_correctly.eq(0)
            # Loss Rule #
            # 0) If we had a correct answer in the first place and is now false -> Reward 1
            # 1) If we had a correct answer in the first place and it is still correct -> Reward 0
            # 2) If we had a false answer in the first place -> Reward 0
            # 3) If the state is invalid - Reward - 1
            # for i in range(0, self.n_arms):
            #     self.rewards[:, i] = 1 - ((self.answers.squeeze(1) - arm_predictions[i]).eq(0) * 1.0).numpy()
            self.rewards = (model_answered_correctly * answer_stayed_the_same * (
                    1.0 - 1.0 * (arm_predictions_before - arm_predictions_after).eq(0))).numpy()
            # + ((1 - answer_stayed_the_same) * np.ones_like(self.rewards) * -1.0).numpy()

            return self.rewards

    def _seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)



def linUCBexperiment(args):
    if osp.exists(f'./results/experiment_linucb'):
        pass
    else:
        os.mkdir(f'./results/experiment_linucb')
    T = 256
    model, model_fool, loader = get_fool_model(device=args.device, load_from=args.load_from,
                                               scenes_path=args.scenes_path, questions_path=args.questions_path,
                                               clvr_path=args.clvr_path,
                                               use_cache=args.use_cache, use_hdf5=args.use_hdf5, batch_size=T)
    test_loader = get_test_loader(load_from=args.load_from,
                                  scenes_path=args.scenes_path, questions_path=args.questions_path,
                                  clvr_path=args.clvr_path)

    with open(f'./results_linucb_{args.scale}.log', 'w+') as fout:
        ### Experiment 1 ###
        train_duration = 50  # X 256 = 149_000
        test_duration = 10_000  # X 1 = 5000
        cls = ContextualStatefulBandit(testbed_model=model, testbed_loader=loader, T=T, n_arms=80,
                                       confusion_model=model_fool, augmentation_strength=args.scale)
        gg = LinUCB(cls,
                    reg_factor=1,
                    delta=0.1,
                    confidence_scaling_factor=1.0,
                    save_path='./results/experiment_linucb/',
                    load_from=f'./results/experiment_linucb/linucb_model_scale_{args.scale}.pt'
                    )

        gg.run(epochs=train_duration, save_every_epochs=50, postfix=f'scale_{args.scale}')


        test_loader_iter = iter(test_loader)
        example_index = 0
        accuracy_drop = 0.0
        while example_index < test_duration:
            try:
                goo = next(test_loader_iter)
                test_features = cls.reset_features(goo)
                ucb, _, action = gg.test(test_features)
                test_rewards = cls.reset_rewards(specific_action=action)
                accuracy_drop += max(0, test_rewards[0][0])
                example_index += 1
                if example_index % 100 == 0 and example_index > 0:
                    _print(f"Scale {args.scale} | Accuracy Dropped By: {100 * (accuracy_drop / example_index)}%")
            except StopIteration:
                break
        _print(f"Scale {args.scale} | Accuracy Dropped By: {100 * (accuracy_drop / example_index)}%")
        fout.write(f"Scale {args.scale} | Accuracy Dropped By: {100 * (accuracy_drop / example_index)}%\n")


def neuralUCBexperiment(args):
    if osp.exists(f'./results/experiment_neuralucb'):
        pass
    else:
        os.mkdir(f'./results/experiment_neuralucb')
    T = 256
    model,model_fool, loader = get_fool_model(device=args.device, load_from=args.load_from,
                                   scenes_path=args.scenes_path, questions_path=args.questions_path,
                                   clvr_path=args.clvr_path,
                                   use_cache=args.use_cache, use_hdf5=args.use_hdf5, batch_size=T)
    test_loader = get_test_loader(load_from=args.load_from,
                                  scenes_path=args.scenes_path, questions_path=args.questions_path,
                                  clvr_path=args.clvr_path)

    with open(f'./results_neuralucb_{args.scale}.log', 'w+') as fout:
        train_duration = 200  # X Batch Size = 128_000
        test_duration = 5000  # X 1 = 5000
        cls = ContextualStatefulBandit(model, loader, T, 80, model_fool, augmentation_strength=args.scale)
        gg = NeuralUCB(cls,
                       hidden_size=20,
                       reg_factor=1.0,
                       delta=0.1,
                       confidence_scaling_factor=1.0,
                       training_window=T // 2,
                       p=0.2,
                       learning_rate=0.01,
                       epochs=50,
                       train_every=(T // 4) - 1,
                       save_path='./results/experiment_neuralucb/'
                       )

        gg.run(epochs=train_duration, save_every_epochs=100, postfix=f'scale_{args.scale}')

        test_loader_iter = iter(test_loader)
        example_index = 0
        accuracy_drop = 0.0
        while example_index < test_duration:
            try:
                goo = next(test_loader_iter)
                test_features = cls.reset_features(goo)
                test_rewards = cls.reset_rewards()
                ucb, _, action = gg.test(test_features)
                accuracy_drop += max(0, test_rewards[0, action])
                example_index += 1
                if example_index % 5 == 0 and example_index > 0:
                    print(f"Scale {args.scale} | Accuracy Dropped By: {100 * (accuracy_drop / example_index)}%")
            except StopIteration:
                break
        print(f"Scale {args.scale} | Accuracy Dropped By: {100 * (accuracy_drop / test_duration)}%")
        fout.write(f"Scale {args.scale} | Accuracy Dropped By: {100 * (accuracy_drop / test_duration)}%\n")


def linUCBexperiment_test(args):
    if osp.exists(f'./results/experiment_linucb'):
        pass
    else:
        os.mkdir(f'./results/experiment_linucb')

    T = 1
    model, model_fool, loader = get_fool_model(device=args.device, load_from=args.load_from,
                                               scenes_path=args.scenes_path, questions_path=args.questions_path,
                                               clvr_path=args.clvr_path,
                                               use_cache=args.use_cache, use_hdf5=args.use_hdf5, batch_size=T)
    test_duration = 10_000  # X 1 = 10_000

    if args.scale == 1 or args.scale == 1.0:
        scale = 1.0
        scale_name = '1.0'
    elif args.scale == 0.5:
        scale = 0.5
        scale_name = '0.5'
    elif args.scale == 0.1:
        scale = 0.1
        scale_name = '0.1'
    cls = ContextualStatefulBandit(testbed_model=model, testbed_loader=loader, T=T, n_arms=80,
                                   confusion_model=None, augmentation_strength=scale)
    gg = LinUCB(cls,
                reg_factor=1.0,
                delta=0.1,
                confidence_scaling_factor=1.0,
                save_path='./results/experiment_linucb/',
                load_from=f'./results/experiment_linucb/linucb_model_scale_{scale_name}.pt'
                )

    test_loader_iter = iter(loader)
    example_index = 0
    accuracy_drop = 0.0
    while example_index < test_duration:
        try:
            goo = next(test_loader_iter)
            test_features = cls.reset_features(goo)
            test_rewards = cls.reset_rewards()
            ucb, _, action = gg.test(test_features)
            accuracy_drop += max(0, test_rewards[0, action])
            example_index += 1
            if example_index % 100 == 0 and example_index > 0:
                _print(f"Scale {args.scale} | Accuracy Dropped By: {100 * (accuracy_drop / example_index)}%")
        except StopIteration:
            break
    _print(f"Scale {args.scale} | Accuracy Dropped By: {100 * (accuracy_drop / example_index)}%")


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
    parser.add_argument('--scale', type=float, help='scale of arguments', default=1.0)
    # parser.add_argument('--load_from', type=str, help='where to load a model', default=None)
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
    elif args.mode == ' neural':
        neuralUCBexperiment(args)
    elif args.mode == 'linear_test':
        linUCBexperiment_test(args)
