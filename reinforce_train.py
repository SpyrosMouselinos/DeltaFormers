import ast
import json
import os
import os.path as osp
import sys

import numpy as np
import torch.nn
import yaml

sys.path.insert(0, osp.abspath('.'))

import argparse
from torch.utils.data import Dataset
from modules.embedder import *
from utils.train_utils import StateCLEVR, ImageCLEVR, ImageCLEVR_HDF5, MixCLEVR_HDF5
from oracle.Oracle_CLEVR import Oracle
import matplotlib.pyplot as plt
import seaborn as sns
from reinforce_modules.policy_networks import Re1nforceTrainer, ImageNetPolicyNet, EmptyNetwork, PolicyNet

sns.set_style('darkgrid')


def _print(something):
    print(something, flush=True)
    return


AVAILABLE_DATASETS = {
    'DeltaRN': [StateCLEVR],
    'DeltaSQFormer': [StateCLEVR],
    'DeltaQFormer': [StateCLEVR],
    'DeltaSQFormerCross': [StateCLEVR, MixCLEVR_HDF5],
    'DeltaSQFormerDisentangled': [StateCLEVR],
    'DeltaSQFormerLinear': [StateCLEVR, MixCLEVR_HDF5],
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
    q2index = vocabs['question_token_to_idx']
    back_translation = vocabs['answer_token_to_idx']
    back_translation.update({'true': back_translation['yes']})
    back_translation.update({'false': back_translation['no']})
    back_translation.update({'__invalid__': 3})  # So after -4 goes to -1

index2q = {v: k for k, v in q2index.items()}
index2a = {v: k for k, v in back_translation.items()}


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
        object_positions = example_state['object_positions'][i].numpy()[number_of_objects] * np.array(
            [3, 3, 360])  # Don't forget that you have scaled them
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
                   use_hdf5=False, mode='state'):
    if device == 'cuda':
        device = 'cuda:0'

    experiment_name = load_from.split('results/')[-1].split('/')[0]
    config = f'./results/{experiment_name}/config.yaml'
    with open(config, 'r') as fin:
        config = yaml.load(fin, Loader=yaml.FullLoader)

    if mode == 'state':
        model = AVAILABLE_MODELS[config['model_architecture']](config)
        _print(f"Loading Model of type: {config['model_architecture']}\n")
        model = load(path=load_from, model=model)
    elif mode == 'imagenet':
        model = EmptyNetwork()
    else:
        raise NotImplementedError

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
    if mode == 'state':
        data_choice = 0
    else:
        data_choice = 1
    val_set = AVAILABLE_DATASETS[config['model_architecture']][data_choice](config=config, split='val',
                                                                            clvr_path=clvr_path,
                                                                            questions_path=questions_path,
                                                                            scenes_path=scenes_path,
                                                                            use_cache=use_cache,
                                                                            return_program=True)

    val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                                 num_workers=0, shuffle=bool(args.mode == 'state'), drop_last=True)

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


def bird_eye_view(index, x, y, mode='before', q=None, a=None):
    if mode == 'before':
        object_positions = x['object_positions'][0].numpy()
        object_colors = [translation_tables['reverse_translation_color'][f] if f != 0 else None for f in
                         x['object_colors'][0].numpy()]
        object_shapes = [translation_tables['reverse_translation_shape'][f] if f != 0 else None for f in
                         x['object_shapes'][0].numpy()]
        object_materials = [translation_tables['reverse_translation_material'][f] if f != 0 else None for f in
                            x['object_materials'][0].numpy()]
        object_sizes = [translation_tables['reverse_translation_size'][f] if f != 0 else None for f in
                        x['object_sizes'][0].numpy()]
        q = [index2q[f] for f in x['question'][0].numpy()]
        a = index2a[y.item() + 4]
    else:
        object_positions = [f['3d_coords'] for f in x['objects']]
        object_colors = [f['color'] for f in x['objects']]
        object_shapes = [f['shape'] for f in x['objects']]
        object_materials = [f['material'] for f in x['objects']]
        object_sizes = [f['size'] for f in x['objects']]
        if q is None:
            q = ''
        else:
            q = [index2q[f] for f in q]
        if a is None:
            a = ''
        else:
            a = index2a[a.item() + 4]

    mmt = {
        'cube': 's',
        'cylinder': 'h',
        'sphere': 'o',
    }

    mst = {
        'large': 8,
        'small': 6
    }
    plt.figure(figsize=(10, 10))
    plt.title(a)
    made_title = ' '.join([f for f in q if (f != '<NULL>') and (f != '<START>') and (f != '<END>')])
    made_title = made_title[:len(made_title) // 2] + '\n' + made_title[len(made_title) // 2:]
    plt.suptitle(f"{made_title}")
    for oi in range(0, 10):
        try:
            x = object_positions[oi][0]
            y = object_positions[oi][1]
        except IndexError:
            continue
        if x != 0 and y != 0:
            plt.scatter(x=x, y=y, c=object_colors[oi], s=mst[object_sizes[oi]] ** 3, marker=mmt[object_shapes[oi]])
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.savefig(f"C:\\Users\\Guldan\\Desktop\\mismatch\\{mode}_{index}.png")
    plt.show()
    plt.close()
    return


class ConfusionGame:
    def __init__(self, testbed_model,
                 confusion_model=None,
                 device='cuda',
                 batch_size=256,
                 confusion_weight=10.0,
                 change_weight=1.0,
                 fail_weight=-1.0,
                 invalid_weight=0.0,
                 mode='state'
                 ):

        self.confusion_weight = confusion_weight
        self.change_weight = change_weight
        self.fail_weight = fail_weight
        self.invalid_weight = invalid_weight
        self.batch_size = batch_size
        self.testbed_model = testbed_model
        self.testbed_model.eval()
        if confusion_model is None:
            self.confusion_model = self.testbed_model
        else:
            self.confusion_model = confusion_model
            self.confusion_model.eval()
        self.device = device
        self.oracle = Oracle(metadata_path='./metadata.json')
        self.mode = mode

    def extract_features(self, iter_on_data):
        """
            Run a Batch Sized - Step in the Confusion Game
            iter_on_data: An iterator on a Dataset
        """
        with torch.no_grad():
            data, y_real, programs = next(iter_on_data)
            if self.mode != 'state':
                data, image_data = data
                self.org_image_data = image_data
            self.org_data = data
            self.org_answers = y_real
            self.programs = programs
            if self.mode != 'state':
                image_data = kwarg_dict_to_device(image_data, self.device)
                data = kwarg_dict_to_device(data, self.device)
                if self.mode == 'imagenet':
                    y_feats = image_data['image']
                else:
                    _, _, y_feats = self.testbed_model(**image_data)
            else:
                data = kwarg_dict_to_device(data, self.device)
                _, _, y_feats = self.testbed_model(**data)

        self.features = y_feats
        self.n_features = self.features.shape[-1]
        return self.features, data, y_real

    @staticmethod
    def alter_object_positions_on_action(features, action_vector):
        object_mask = features['types'][:, :10]
        object_positions = features['object_positions']
        av = action_vector.detach().cpu()
        # 10 first items are x perturbations #
        # 10 next items are y perturbations #
        x_change = av[:, :10] * object_mask
        x_change = x_change.unsqueeze(-1)
        y_change = av[:, 10:] * object_mask
        y_change = y_change.unsqueeze(-1)
        # Build it in BS X 3 format
        z_change = torch.zeros_like(y_change)
        change = torch.cat([x_change, y_change, z_change], dim=2)
        object_positions += change
        return object_positions

    def step(self, action_vector):
        validity = []
        predictions_after = []
        predictions_before = self.confusion_model(**kwarg_dict_to_device(self.org_data, self.device))[
            0].detach().cpu().argmax(1)
        programs = translate_str_program(self.programs)
        object_positions = self.alter_object_positions_on_action(self.org_data, action_vector)
        self.org_data['object_positions'] = object_positions
        scene = translate_state(self.org_data)
        res = self.oracle(programs, scene, None)
        res = translate_answer(res)
        predictions_after.append(
            self.confusion_model(**kwarg_dict_to_device(self.org_data, self.device))[0].detach().cpu().argmax(1))
        validity.append(res)

        return predictions_after, predictions_before, validity, scene

    def get_rewards(self, action_vector):
        predictions_after, predictions_before, validity, scene = self.step(action_vector=action_vector)
        predictions_before = predictions_before.unsqueeze(1)
        predictions_after = torch.stack(predictions_after, dim=1).long()
        if self.batch_size > 1:
            validity = [torch.LongTensor(f) for f in validity]
        answer_stayed_the_same = self.org_answers - torch.stack(validity, dim=1).long()
        answer_stayed_the_same = 1.0 * answer_stayed_the_same.eq(0)
        model_answered_correctly = self.org_answers - predictions_before
        model_answered_correctly = 1.0 * model_answered_correctly.eq(0)

        confusion_rewards = (model_answered_correctly * answer_stayed_the_same * (
                1.0 - 1.0 * (predictions_before - predictions_after).eq(0)))

        change_rewards = (answer_stayed_the_same * (
                1.0 - 1.0 * (predictions_before - predictions_after).eq(0)))

        fail_rewards = self.fail_weight * torch.ones_like(change_rewards)
        invalid_scene_rewards = self.invalid_weight * (1 - answer_stayed_the_same)
        self.rewards = self.confusion_weight * confusion_rewards.numpy() + (
            self.change_weight) * change_rewards.numpy() + fail_rewards.numpy() + invalid_scene_rewards.numpy()
        return self.rewards, confusion_rewards, change_rewards, scene, predictions_after


def PolicyEvaluation(args):
    if osp.exists(f'./results/experiment_reinforce'):
        pass
    else:
        os.mkdir(f'./results/experiment_reinforce')
    BS = args.bs
    model, model_fool, loader = get_fool_model(device=args.device, load_from=args.load_from,
                                               scenes_path=args.scenes_path, questions_path=args.questions_path,
                                               clvr_path=args.clvr_path,
                                               use_cache=args.use_cache, use_hdf5=args.use_hdf5, batch_size=BS, mode=args.mode)

    train_duration = args.train_duration
    rl_game = ConfusionGame(testbed_model=model, confusion_model=model_fool, device='cuda', batch_size=BS,
                            confusion_weight=args.confusion_weight, change_weight=args.change_weight,
                            fail_weight=args.fail_weight, invalid_weight=args.invalid_weight, mode=args.mode)
    if args.mode == 'state' or args.mode == 'visual':
        model = PolicyNet(input_size=128, hidden_size=256, dropout=0.0, reverse_input=True)
    elif args.mode == 'imagenet':
        model = ImageNetPolicyNet(input_size=128, hidden_size=256, dropout=0.0, reverse_input=True)
    else:
        raise ValueError
    if args.cont > 0:
        print("Loading model...")
        model.load('./results/experiment_reinforce/model_reinforce.pt')
    trainer = Re1nforceTrainer(model=model, game=rl_game, dataloader=loader, device=args.device, lr=args.lr,
                               train_duration=train_duration, batch_size=BS)

    trainer.train(log_every=5, save_every=10000)
    # trainer.evaluate()


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
    parser.add_argument('--confusion_weight', type=float, help='what kind of experiment to run', default=10.0)
    parser.add_argument('--change_weight', type=float, help='what kind of experiment to run', default=0.0)
    parser.add_argument('--fail_weight', type=float, help='what kind of experiment to run', default=-1.0)
    parser.add_argument('--invalid_weight', type=float, help='what kind of experiment to run', default=-10.0)
    parser.add_argument('--train_duration', type=int, help='what kind of experiment to run', default=20000)
    parser.add_argument('--lr', type=float, help='what kind of experiment to run', default=0.001)
    parser.add_argument('--bs', type=int, help='what kind of experiment to run', default=256)
    parser.add_argument('--cont', type=int, help='what kind of experiment to run', default=0)
    parser.add_argument('--mode', type=str, help='state | visual | imagenet', default='state')

    args = parser.parse_args()
    PolicyEvaluation(args)
