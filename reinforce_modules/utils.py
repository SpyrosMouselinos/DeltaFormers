import ast
import copy
import itertools
import json
import math
import os
import os.path as osp
import sys
import numpy as np
import torch.nn
import yaml
from torch import optim
from utils.train_utils import SGD_GC
from fool_models.mdter_utils import load_mdetr, inference_with_mdetr
from fool_models.rnfp_utils import load_rnfp, inference_with_rnfp, load_loader as load_loader_rnfp
from fool_models.tbd_utils import load_resnet_backbone as load_resnet_tbd_backbone
from fool_models.tbd_utils import load_tbd, inference_with_tbh

sys.path.insert(0, osp.abspath('.'))

from torch.utils.data import Dataset
from modules.embedder import *
from utils.train_utils import StateCLEVR, ImageCLEVR, ImageCLEVR_HDF5, MixCLEVR_HDF5
import matplotlib.pyplot as plt
import seaborn as sns
from reinforce_modules.policy_networks import EmptyNetwork, PolicyNet
from fool_models.iep_utils import load_iep, inference_with_iep
from fool_models.film_utils import load_film, load_resnet_backbone, inference_with_film
from fool_models.stack_attention_utils import load_cnn_sa, inference_with_cnn_sa
from neural_render.blender_render_utils.helpers import render_image
from skimage.color import rgba2rgb
from skimage.io import imread
from skimage.transform import resize as imresize
from oracle.Oracle_CLEVR import Oracle

sns.set_style('darkgrid')


def _print(something):
    print(something, flush=True)
    return


def accuracy_metric(y_pred, y_true):
    acc = (y_pred.argmax(1) == y_true).float().detach().cpu().numpy()
    return float(100 * acc.sum() / len(acc))


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

    batch_size = example_state['types'].cpu().numpy().shape[0]

    for i in range(batch_size):
        template = header_template()
        number_of_objects = np.where(example_state['types'][i].cpu().numpy() == 1)[0]
        object_positions = example_state['object_positions'][i].cpu().numpy()[number_of_objects] * np.array(
            [3, 3, 360])  # Don't forget that you have scaled them
        object_colors = example_state['object_colors'][i].cpu().numpy()[number_of_objects]
        object_shapes = example_state['object_shapes'][i].cpu().numpy()[number_of_objects]
        object_materials = example_state['object_materials'][i].cpu().numpy()[number_of_objects]
        object_sizes = example_state['object_sizes'][i].cpu().numpy()[number_of_objects]
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
    try:
        if torch.cuda.device_count() >= 1 and any(
                k.startswith('module.') for k in checkpoint['model_state_dict'].keys()):
            checkpoint['model_state_dict'] = {k.replace('module.', ''): v for k, v in
                                              checkpoint['model_state_dict'].items()}
    except KeyError:
        if torch.cuda.device_count() >= 1 and any(k.startswith('module.') for k in checkpoint.keys()):
            checkpoint = {k.replace('module.', ''): v for k, v in
                          checkpoint.items()}
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except KeyError:
        model.load_state_dict(checkpoint)

    try:
        _print(f"Your model achieves {round(checkpoint['val_loss'], 4)} validation loss\n")
    except:
        _print("No Loss registered")
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
                   use_hdf5=False, mode='state', effective_range=None, mos_epoch=164, randomize_range=False):
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
    # config_fool = f'./results/experiment_linear_sq/config.yaml'
    config_fool = f'./results/experiment_rn/config.yaml'
    with open(config_fool, 'r') as fin:
        config_fool = yaml.load(fin, Loader=yaml.FullLoader)

    model_fool = AVAILABLE_MODELS[config_fool['model_architecture']](config_fool)
    _print(f"Loading Model of type: {config_fool['model_architecture']}\n")
    # model_fool = load(path=f'./results/experiment_linear_sq/model.pt', model=model_fool)
    model_fool = load(path=f'./results/experiment_rn/mos_epoch_{mos_epoch}.pt', model=model_fool)
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
                                                                            return_program=True,
                                                                            effective_range=effective_range,
                                                                            randomize_range=randomize_range)

    val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                                 num_workers=0, shuffle=False, drop_last=False)

    _print(f"Loader has : {len(val_dataloader)} batches\n")
    return model, model_fool, val_dataloader


def get_visual_fool_model(device, load_from=None, clvr_path='data/', questions_path='data/', scenes_path='data/',
                          use_cache=False, batch_size=128,
                          use_hdf5=False, mode='state', effective_range=None, fool_model=None, randomize_range=False, range_offset=None):
    if fool_model is None:
        _print("No model is chosen...aborting")
        return
    elif fool_model == 'sa':
        resnet = load_resnet_backbone()
        model_fool = load_cnn_sa()
        output_shape = 224
    elif fool_model == 'film':
        resnet = load_resnet_backbone()
        model_fool = load_film()
        output_shape = 224
    elif fool_model == 'iep':
        resnet = load_resnet_backbone()
        model_fool = load_iep()
        output_shape = 224
    elif fool_model == 'rnfp':
        resnet = None
        model_fool = load_rnfp()
        output_shape = 128
    elif fool_model == 'tbd':
        resnet = load_resnet_tbd_backbone()
        model_fool = load_tbd()
        output_shape = 224
    elif fool_model == 'mdetr':
        resnet = None
        model_fool = load_mdetr()
        output_shape = 224
    else:
        raise NotImplementedError

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
    elif mode == 'visual':
        model = AVAILABLE_MODELS[config['model_architecture']](config)
        _print(f"Loading Model of type: {config['model_architecture']}\n")
        model = load(path=load_from, model=model)
    else:
        raise NotImplementedError

    model = model.to(device)
    model.eval()

    _print(f"Loading Model of type: {fool_model}\n")

    val_set = MixCLEVR_HDF5(config=config, split='val',
                            clvr_path=clvr_path,
                            questions_path=questions_path,
                            scenes_path=scenes_path,
                            use_cache=use_cache,
                            return_program=True,
                            effective_range=effective_range, output_shape=output_shape, randomize_range=randomize_range, effective_range_offset=range_offset)
    initial_example = val_set.effective_range_offset
    val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                                 num_workers=0, shuffle=False, drop_last=False)
    _print(f"Loader has : {len(val_dataloader)} batches\n")
    _print(f"Calculating and storing answers of {fool_model} model!\n")
    if fool_model == 'sa':
        predictions_before_pre_calc = inference_with_cnn_sa(val_dataloader, model_fool, resnet)
    elif fool_model == 'film':
        predictions_before_pre_calc = inference_with_film(val_dataloader, model_fool, resnet)
    elif fool_model == 'iep':
        predictions_before_pre_calc = inference_with_iep(val_dataloader, model_fool, resnet)
    elif fool_model == 'rnfp':
        predictions_before_pre_calc = inference_with_rnfp(val_dataloader, model_fool, None)
    elif fool_model == 'tbd':
        predictions_before_pre_calc = inference_with_tbh(val_dataloader, model_fool, resnet)
    elif fool_model == 'mdetr':
        predictions_before_pre_calc = inference_with_mdetr(val_dataloader, model_fool, None)
    else:
        raise NotImplementedError

    return model, (model_fool, resnet), val_dataloader, predictions_before_pre_calc, initial_example


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


def get_defense_models(adversarial_agent_load_from=None,
                       feature_extractor_load_from=None,
                       vqa_model_load_type=None,
                       batch_size=128,
                       effective_range=None,
                       effective_range_offset=None,
                       randomize_range=False):
    # Loading VQA Agent #
    _print(f"Loading VQA Agent: {vqa_model_load_type}\n")
    if vqa_model_load_type is None:
        _print("No model is chosen...aborting")
        return
    elif vqa_model_load_type == 'film':
        resnet = load_resnet_backbone()
        vqa_agent = load_film()
        output_shape = 224
    elif vqa_model_load_type == 'rnfp':
        resnet = None
        vqa_agent = load_rnfp()
        output_shape = 128
    else:
        raise NotImplementedError

    device = 'cuda:0'
    # Loading Mini Game #
    minigames = []
    if isinstance(effective_range_offset, list):
        for r in effective_range_offset:
            val_set = MixCLEVR_HDF5(config=None, split='val',
                                    clvr_path='./data',
                                    questions_path='./data',
                                    scenes_path='./data',
                                    use_cache=False,
                                    return_program=True,
                                    effective_range=effective_range, output_shape=output_shape,
                                    randomize_range=randomize_range, effective_range_offset=r)

            minigame = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                                   num_workers=0, shuffle=False, drop_last=False)
            minigames.append(minigame)
            _print(f"Minigame has : {len(minigame)} batches\n")
    else:
        val_set = MixCLEVR_HDF5(config=None, split='val',
                                clvr_path='./data',
                                questions_path='./data',
                                scenes_path='./data',
                                use_cache=False,
                                return_program=True,
                                effective_range=effective_range, output_shape=output_shape,
                                randomize_range=randomize_range, effective_range_offset=effective_range_offset)

        minigame = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                               num_workers=0, shuffle=False, drop_last=False)
        minigames.append(minigame)
        _print(f"Minigame has : {len(minigame)} batches\n")
        minigames.append(None)

    # Loading Feature Extractor #
    experiment_name = feature_extractor_load_from.split('results/')[-1].split('/')[0]
    config = f'./results/{experiment_name}/config.yaml'
    with open(config, 'r') as fin:
        config = yaml.load(fin, Loader=yaml.FullLoader)

    feature_extractor = AVAILABLE_MODELS[config['model_architecture']](config)
    _print(f"Loading Feature Extractor: {config['model_architecture']}\n")
    feature_extractor = load(path=feature_extractor_load_from, model=feature_extractor)

    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()

    # Loading Adversarial Agent #
    _print("Loading Adversarial Agent...")
    adversarial_agents = []
    if isinstance(adversarial_agent_load_from, list):
        for entry in adversarial_agent_load_from:
            adversarial_agent = PolicyNet(input_size=128, hidden_size=256, dropout=0.0, reverse_input=True)
            adversarial_agent.load(entry)
            adversarial_agents.append(adversarial_agent)
    else:
        adversarial_agent = PolicyNet(input_size=128, hidden_size=256, dropout=0.0, reverse_input=True)
        adversarial_agent.load(adversarial_agent_load_from)
        adversarial_agents = [adversarial_agent, None]

    return adversarial_agents, feature_extractor, vqa_agent, resnet, minigames


class ConfusionGame:
    def __init__(self, testbed_model,
                 confusion_model=None,
                 device='cuda',
                 batch_size=256,
                 confusion_weight=10.0,
                 change_weight=1.0,
                 fail_weight=-1.0,
                 invalid_weight=0.0,
                 mode='state',
                 render=False
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
            if isinstance(confusion_model, tuple):
                a = confusion_model[0].to('cuda')
                a.eval()
                b = confusion_model[1].to('cuda')
                b.eval()
                self.confusion_model = (a, b)
            else:
                self.confusion_model = confusion_model
                self.confusion_model.eval()
        self.device = device
        self.oracle = Oracle(metadata_path='./metadata.json')
        self.mode = mode
        self.render = render

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
                    _, _, y_feats = self.testbed_model(**data)
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
        if not isinstance(action_vector, np.ndarray):
            av = action_vector.detach().cpu()
        else:
            av = torch.FloatTensor(action_vector)
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
        return object_positions.clip(-1.0, 1.0)

    @staticmethod
    def render_check(img_xs, img_ys, img_sizes, img_shapes):
        directions = {'below': [-0.0, -0.0, -1.0],
                      'front': [0.754490315914154, -0.6563112735748291, -0.0],
                      'above': [0.0, 0.0, 1.0],
                      'right': [0.6563112735748291, 0.7544902563095093, -0.0],
                      'behind': [-0.754490315914154, 0.6563112735748291, 0.0],
                      'left': [-0.6563112735748291, -0.7544902563095093, 0.0]}
        positions = []
        assert len(img_xs) == len(img_ys)
        for i in range(len(img_xs)):
            x = img_xs[i]
            y = img_ys[i]
            r = img_sizes[i]
            if r == 0:
                r = 0.7
            else:
                r = 0.3
            s = img_shapes[i]
            if s == 0:
                r /= math.sqrt(2)
            dists_good = True
            margins_good = True
            for (xx, yy, rr) in positions:
                dx, dy = x - xx, y - yy
                dist = math.sqrt(dx * dx + dy * dy)
                if dist - r - rr < 0.25:
                    dists_good = False
                    break
                for direction_name in ['left', 'right', 'front', 'behind']:
                    direction_vec = directions[direction_name]
                    assert direction_vec[2] == 0
                    margin = dx * direction_vec[0] + dy * direction_vec[1]
                    if 0 < margin < 0.4:
                        margins_good = False
                        break
                if not margins_good:
                    break

            if not dists_good or not margins_good:
                return False
            positions.append((x, y, r))
        return True

    def state2img(self, state, bypass=False, custom_index=0, delete_every=True, retry=False):
        wr = []
        images_to_be_rendered = n_possible_images = state['positions'].size(0)
        n_objects_per_image = state['types'][:, :10].sum(1).numpy()
        key_light_jitter = fill_light_jitter = back_light_jitter = [0.5] * n_possible_images
        if retry:
            choices = [1, 0.5, 0.2, -0.2, -0.5, 0, -1]
        else:
            choices = [1]
        for jitter in choices:
            camera_jitter = [jitter] * n_possible_images
            xs = []
            ys = []
            zs = []
            colors = []
            shapes = []
            materials = []
            sizes = []
            questions = []
            for image_idx in range(n_possible_images):
                tmp_x = []
                tmp_y = []
                tmp_z = []
                tmp_colors = []
                tmp_shapes = []
                tmp_materials = []
                tmp_sizes = []
                for object_idx in range(n_objects_per_image[image_idx]):
                    tmp_x.append(state['object_positions'][image_idx, object_idx].numpy()[0] * 3)
                    tmp_y.append(state['object_positions'][image_idx, object_idx].numpy()[1] * 3)
                    tmp_z.append(state['object_positions'][image_idx, object_idx].numpy()[2] * 360)
                    tmp_colors.append(state['object_colors'][image_idx, object_idx].item() - 1)
                    tmp_shapes.append(state['object_shapes'][image_idx, object_idx].item() - 1)
                    tmp_materials.append(state['object_materials'][image_idx, object_idx].item() - 1)
                    tmp_sizes.append(state['object_sizes'][image_idx, object_idx].item() - 1)
                if self.render_check(tmp_x, tmp_y, tmp_sizes, tmp_shapes) or bypass:
                    xs.append(tmp_x)
                    ys.append(tmp_y)
                    zs.append(tmp_z)
                    colors.append(tmp_colors)
                    shapes.append(tmp_shapes)
                    materials.append(tmp_materials)
                    sizes.append(tmp_sizes)
                    questions.append(state['question'][image_idx])
                    wr.append(image_idx)
                else:
                    # print(f"Check Failed for index {custom_index}!")
                    images_to_be_rendered -= 1

            if delete_every:
                for target in os.listdir('./neural_render/images'):
                    if 'Rendered' in target:
                        try:
                            os.remove('./neural_render/images/' + target)
                        except:
                            pass
            assembled_images = render_image(key_light_jitter=key_light_jitter, fill_light_jitter=fill_light_jitter,
                                            back_light_jitter=back_light_jitter, camera_jitter=camera_jitter,
                                            per_image_x=xs, per_image_y=ys, per_image_theta=zs, per_image_shapes=shapes,
                                            per_image_colors=colors, per_image_sizes=sizes,
                                            per_image_materials=materials,
                                            num_images=images_to_be_rendered, split='Rendered', start_idx=custom_index,
                                            workers=1)
            final_images = []
            final_questions = []
            for fake_idx, (pair, real_idx) in enumerate(zip(assembled_images, wr)):
                is_rendered = pair[1]
                if is_rendered:
                    final_images.append(real_idx)
                    final_questions.append(questions[fake_idx])

            if retry:
                if len(final_images) == 1:
                    return final_images, final_questions
                else:
                    continue
        return final_images, final_questions

    def perpare_and_pass(self, resnet, questions, rendered_images):
        if resnet is None:
            if hasattr(self.confusion_model, 'COLORS'):
                # MDetr Model #
                img_size = (224, 224)
            else:
                # RNFP Model #
                img_size = (128, 128)
        else:
            img_size = (224, 224)
        path = './neural_render/images'
        images = [f'./neural_render/images/{f}' for f in os.listdir(path) if 'Rendered' in f and '.png' in f]
        feat_list = []
        if len(images) == 0:
            return [-1] * self.batch_size
        ### Read the images ###
        for image in images:
            img = imread(image)
            img = rgba2rgb(img)
            img = imresize(img, img_size)
            img = img.astype('float32')
            img = img.transpose(2, 0, 1)[None]
            mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
            std = np.array([0.229, 0.224, 0.224]).reshape(1, 3, 1, 1)
            img = (img - mean) / std
            img_var = torch.FloatTensor(img).to('cuda')
            if resnet is not None:
                ### Pass through Resnet ###
                feat_list.append(resnet(img_var))
            else:
                feat_list.append(img_var)

        ### Stack them with Questions ###
        feats = torch.cat(feat_list, dim=0)
        questions = torch.cat([f.unsqueeze(0) for f in questions], dim=0)
        ### Pass through Model ###
        if isinstance(self.confusion_model, tuple):

            questions = questions.to('cuda')
            feats = feats.to('cuda')
            program_generator, execution_engine = self.confusion_model
            if hasattr(program_generator, 'reinforce_sample'):
                if hasattr(program_generator, 'do_not_trust_reinforce'):
                    progs = []
                    for i in range(questions.size(0)):
                        program = program_generator.reinforce_sample(questions[i, :].view(1, -1))
                        progs.append(program.cpu().numpy().squeeze())
                    programs_pred = torch.LongTensor(np.asarray(progs)).to('cuda')
                else:
                    programs_pred = program_generator.reinforce_sample(
                        questions,
                        temperature=1,
                        argmax=True)
                scores = execution_engine(feats, programs_pred)
            else:
                programs_pred = program_generator(questions)
                scores = execution_engine(feats, programs_pred, save_activations=False)
            _, preds = scores.data.cpu().max(1)
        else:
            questions = questions.to('cuda')
            feats = feats.to('cuda')
            if resnet is not None:
                scores = self.confusion_model(questions, feats)
            else:
                scores, _, _ = self.confusion_model(**{'image': feats, 'question': questions})
            if hasattr(self.confusion_model, 'COLORS'):
                preds = scores - 4
            else:
                _, preds = scores.data.cpu().max(1)
        if resnet is not None:
            # TBD Model #
            if isinstance(self.confusion_model, tuple) and hasattr(program_generator, 'reinforce_sample') and hasattr(
                    program_generator, 'do_not_trust_reinforce'):
                correct_preds = []
                for item in preds.detach().cpu().numpy():
                    correct_preds.append(execution_engine.translate_codes[item] - 4)
                    preds = correct_preds
            else:
                preds = [f - 4 for f in preds]
        else:
            # RNFP Model / MDetr Model#
            pass
        send_back = np.ones(self.batch_size) * (-1)
        k = 0
        for i in range(self.batch_size):
            if i in rendered_images:
                send_back[i] = preds[k]
                k += 1
        return send_back

    def step(self, action_vector, render=False, current_predictions_before=None, resnet=None):

        predictions_after = []
        if render:
            ### Pre calculate predictions before in order to save time ###
            predictions_before = current_predictions_before
        else:
            predictions_before = self.confusion_model(**kwarg_dict_to_device(self.org_data, self.device))[
                0].detach().cpu().argmax(1)

        programs = translate_str_program(self.programs)
        object_positions = self.alter_object_positions_on_action(self.org_data, action_vector)
        self.org_data['object_positions'] = object_positions
        scene = translate_state(self.org_data)
        res = self.oracle(programs, scene, None)
        res = translate_answer(res)
        validity = res
        state_after = self.org_data

        if render:
            rendered_images, questions = self.state2img(self.org_data)
            predictions_before = [f for f in predictions_before]
            scene = [f for f in scene]
            predictions_after.append(self.perpare_and_pass(resnet, questions, rendered_images))
        else:
            predictions_after.append(
                self.confusion_model(**kwarg_dict_to_device(self.org_data, self.device))[0].detach().cpu().argmax(1))

        return predictions_after, predictions_before, validity, scene, state_after

    def get_rewards(self, action_vector, current_predictions_before=None, resnet=None):
        predictions_after, predictions_before, validity, scene, state_after = self.step(action_vector=action_vector,
                                                                                        render=self.render,
                                                                                        current_predictions_before=current_predictions_before,
                                                                                        resnet=resnet
                                                                                        )
        pb = []
        pa = []
        not_rendered = []
        for b, a in list(itertools.zip_longest(predictions_before, predictions_after[0])):
            if b is not None:
                if a == -1:
                    pb.append(int(b))
                    pa.append(int(b))
                    not_rendered.append(1)
                else:
                    pb.append(int(b))
                    pa.append(int(a))
                    not_rendered.append(0)
            else:
                break
        predictions_before, predictions_after = pb, pa
        not_rendered = torch.LongTensor(not_rendered)

        if isinstance(predictions_before, list):
            predictions_before = torch.LongTensor(predictions_before)

        if isinstance(predictions_after, list):
            predictions_after = torch.LongTensor(predictions_after)
        if self.batch_size == 1:
            validity = [validity]
        answer_stayed_the_same = self.org_answers - torch.stack(validity, dim=0).long()
        answer_stayed_the_same = 1.0 * answer_stayed_the_same.eq(0).squeeze(1)
        model_answered_correctly = self.org_answers.squeeze(1) - predictions_before
        model_answered_correctly = 1.0 * model_answered_correctly.eq(0)

        confusion_rewards = (model_answered_correctly * answer_stayed_the_same * (
                1.0 * (predictions_before != predictions_after)))

        change_rewards = (answer_stayed_the_same * (1.0 * (predictions_before != predictions_after)))

        fail_rewards = self.fail_weight * torch.ones_like(change_rewards)
        invalid_scene_rewards = self.invalid_weight * not_rendered
        self.rewards = self.confusion_weight * confusion_rewards.numpy() + self.change_weight * change_rewards.numpy() + fail_rewards.numpy() + invalid_scene_rewards.numpy()
        return self.rewards, confusion_rewards, change_rewards, fail_rewards, invalid_scene_rewards, scene, predictions_after, state_after


class DefenseGame:
    def __init__(self,
                 vqa_model=None,
                 adversarial_agent_1=None,
                 adversarial_agent_2=None,
                 feature_extractor_backbone=None,
                 resnet=None,
                 device='cuda',
                 batch_size=5,
                 defense_rounds=10,
                 pipeline='extrapolation',
                 mode='visual',
                 train_range='0_10',
                 train_name='rnfp'
                 ):
        if feature_extractor_backbone is None:
            raise ValueError('Feature Extractor Should be set to RN-State')
        self.feature_extractor_backbone = feature_extractor_backbone
        self.device = device
        if batch_size > 32:
            raise ResourceWarning(
                f'You have selected Batch Size {batch_size}, this is a bit too much... but continue nevertheless\n')
        self.batch_size = batch_size
        if pipeline.lower() not in ['interpolation', 'extrapolation']:
            raise NotImplementedError(
                f'Pipeline can only be one of the following: [interpolate / extrapolate], you entered {pipeline}\n')
        if isinstance(vqa_model, tuple):
            raise NotImplementedError('Have not made this yet...\n')
        self.vqa_model = vqa_model
        self.vqa_model.to(self.device)
        self.adversarial_agent_1 = adversarial_agent_1
        self.adversarial_agent_1.to(self.device)
        self.adversarial_agent_2 = adversarial_agent_2
        if adversarial_agent_2 is not None:
            self.adversarial_agent_2.to(self.device)
        self.defense_rounds = defense_rounds
        self.oracle = Oracle(metadata_path='./metadata.json')
        self.pipeline = pipeline
        self.mode = mode
        self.resnet = resnet
        self.change_weight = 1
        self.confusion_weight = 0.1
        self.fail_weight = -0.1
        self.invalid_weight = -0.8
        self.skip_rendering = False
        self.train_range = train_range
        self.train_name = train_name

    def save(self, model, range, path):
        if os.path.exists('/'.join(path.split('/')[:-1])):
            pass
        else:
            os.mkdir('/'.join(path.split('/')[:-1]))
        torch.save({
            'model_state_dict': model.state_dict(),
        }, path + f'/defense_against_{range}.pt')
        return

    def stop_rendering(self):
        self.skip_rendering = True

    def start_rendering(self):
        self.skip_rendering = False

    def extract_features(self, iter_on_data):
        """
            Run a Batch Sized - Step in the Defense Game
            iter_on_data: An iterator on a Minigame Dataset
        """
        with torch.no_grad():
            data, y_real, programs = next(iter_on_data)
            if self.mode != 'state':
                data, image_data = data
                image_data = kwarg_dict_to_device(image_data, self.device)
                data = kwarg_dict_to_device(data, self.device)
                _, _, features = self.feature_extractor_backbone(**data)
            else:
                data = kwarg_dict_to_device(data, self.device)
                _, _, features = self.feature_extractor_backbone(**data)

        return features, data, image_data, programs, y_real

    def alter_object_positions_on_action(self, features, action_vector):
        object_mask = features['types'][:, :10]
        object_positions = features['object_positions']
        if not isinstance(action_vector, np.ndarray):
            av = action_vector.detach().to(self.device)
        else:
            av = torch.FloatTensor(action_vector).to(self.device)
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
        return object_positions.clip(-1.0, 1.0)

    @staticmethod
    def render_check(img_xs, img_ys, img_sizes, img_shapes):
        directions = {'below': [-0.0, -0.0, -1.0],
                      'front': [0.754490315914154, -0.6563112735748291, -0.0],
                      'above': [0.0, 0.0, 1.0],
                      'right': [0.6563112735748291, 0.7544902563095093, -0.0],
                      'behind': [-0.754490315914154, 0.6563112735748291, 0.0],
                      'left': [-0.6563112735748291, -0.7544902563095093, 0.0]}
        positions = []
        assert len(img_xs) == len(img_ys)
        for i in range(len(img_xs)):
            x = img_xs[i]
            y = img_ys[i]
            r = img_sizes[i]
            if r == 0:
                r = 0.7
            else:
                r = 0.3
            s = img_shapes[i]
            if s == 0:
                r /= math.sqrt(2)
            dists_good = True
            margins_good = True
            for (xx, yy, rr) in positions:
                dx, dy = x - xx, y - yy
                dist = math.sqrt(dx * dx + dy * dy)
                if dist - r - rr < 0.25:
                    dists_good = False
                    break
                for direction_name in ['left', 'right', 'front', 'behind']:
                    direction_vec = directions[direction_name]
                    assert direction_vec[2] == 0
                    margin = dx * direction_vec[0] + dy * direction_vec[1]
                    if 0 < margin < 0.4:
                        margins_good = False
                        break
                if not margins_good:
                    break

            if not dists_good or not margins_good:
                return False
            positions.append((x, y, r))
        return True

    def state2img(self, state, bypass=False, custom_index=0, delete_every=True, retry=False, split='Rendered'):
        wr = []
        images_to_be_rendered = n_possible_images = state['positions'].size(0)
        n_objects_per_image = state['types'][:, :10].sum(1).cpu().numpy()
        key_light_jitter = fill_light_jitter = back_light_jitter = [0.5] * n_possible_images
        if retry:
            choices = [1, 0.5, 0.2, -0.2, -0.5, 0, -1]
        else:
            choices = [1]
        for jitter in choices:
            camera_jitter = [jitter] * n_possible_images
            xs = []
            ys = []
            zs = []
            colors = []
            shapes = []
            materials = []
            sizes = []
            questions = []
            for image_idx in range(n_possible_images):
                tmp_x = []
                tmp_y = []
                tmp_z = []
                tmp_colors = []
                tmp_shapes = []
                tmp_materials = []
                tmp_sizes = []
                for object_idx in range(n_objects_per_image[image_idx]):
                    tmp_x.append(state['object_positions'][image_idx, object_idx].cpu().numpy()[0] * 3)
                    tmp_y.append(state['object_positions'][image_idx, object_idx].cpu().numpy()[1] * 3)
                    tmp_z.append(state['object_positions'][image_idx, object_idx].cpu().numpy()[2] * 360)
                    tmp_colors.append(state['object_colors'][image_idx, object_idx].cpu().item() - 1)
                    tmp_shapes.append(state['object_shapes'][image_idx, object_idx].cpu().item() - 1)
                    tmp_materials.append(state['object_materials'][image_idx, object_idx].cpu().item() - 1)
                    tmp_sizes.append(state['object_sizes'][image_idx, object_idx].cpu().item() - 1)
                if self.render_check(tmp_x, tmp_y, tmp_sizes, tmp_shapes) or bypass:
                    xs.append(tmp_x)
                    ys.append(tmp_y)
                    zs.append(tmp_z)
                    colors.append(tmp_colors)
                    shapes.append(tmp_shapes)
                    materials.append(tmp_materials)
                    sizes.append(tmp_sizes)
                    questions.append(state['question'][image_idx])
                    wr.append(image_idx)
                else:
                    # print(f"Check Failed for index {custom_index}!")
                    images_to_be_rendered -= 1

            if delete_every:
                for target in os.listdir('./neural_render/images'):
                    if 'Rendered' in target:
                        try:
                            os.remove('./neural_render/images/' + target)
                        except:
                            pass

            assembled_images = render_image(key_light_jitter=key_light_jitter, fill_light_jitter=fill_light_jitter,
                                            back_light_jitter=back_light_jitter, camera_jitter=camera_jitter,
                                            per_image_x=xs, per_image_y=ys, per_image_theta=zs, per_image_shapes=shapes,
                                            per_image_colors=colors, per_image_sizes=sizes,
                                            per_image_materials=materials,
                                            num_images=images_to_be_rendered, split=split, start_idx=custom_index,
                                            workers=1)
            final_images = []
            final_questions = []
            for fake_idx, (pair, real_idx) in enumerate(zip(assembled_images, wr)):
                is_rendered = pair[1]
                if is_rendered:
                    final_images.append(custom_index + real_idx)
                    final_questions.append(questions[fake_idx])

            if retry:
                if len(final_images) == 1:
                    return final_images, final_questions
                else:
                    continue
        return final_images, final_questions

    def perpare_and_pass(self, vqa_model, resnet, questions, rendered_images, id_list=None, split='Rendered'):
        def add_nulls2(int, cnt):
            nulls = str(int)
            for i in range(cnt - len(str(int))):
                nulls = '0' + nulls
            return nulls

        if resnet is None:
            img_size = (128, 128)
        else:
            img_size = (224, 224)
        path = './neural_render/images'
        images = [f'./neural_render/images/{f}' for f in os.listdir(path) if split in f and '.png' in f]
        if id_list is None:
            pass
        else:
            final_images = []
            for image in images:
                for id_code in id_list:
                    id_code = add_nulls2(id_code, 6)
                    if id_code in image:
                        final_images.append(image)
            images = final_images
        feat_list = []
        if len(images) == 0:
            return [-1] * self.batch_size, None
        ### Read the images ###
        for image in images:
            img = imread(image)
            img = rgba2rgb(img)
            img = imresize(img, img_size)
            img = img.astype('float32')
            img = img.transpose(2, 0, 1)[None]
            mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
            std = np.array([0.229, 0.224, 0.224]).reshape(1, 3, 1, 1)
            img = (img - mean) / std
            img_var = torch.FloatTensor(img).to('cuda')
            if resnet is not None:
                ### Pass through Resnet ###
                feat_list.append(resnet(img_var))
            else:
                feat_list.append(img_var)

        ### Stack them with Questions ###
        feats = torch.cat(feat_list, dim=0)
        questions = torch.cat([f.unsqueeze(0) for f in questions], dim=0)
        ### Pass through Model ###
        questions = questions.to('cuda')
        feats = feats.to('cuda')
        if resnet is not None:
            scores = vqa_model(questions, feats)
        else:
            scores, _, _ = vqa_model(**{'image': feats, 'question': questions})

        _, preds = scores.data.cpu().max(1)
        if resnet is not None:
            preds = [f - 4 for f in preds]
        else:
            # RNFP Model / MDetr Model#
            pass

        send_back = np.ones(self.batch_size) * (-1)
        k = 0
        for i in range(self.batch_size):
            if i in rendered_images:
                send_back[i] = preds[k]
                k += 1
        return send_back, scores

    def step(self, vqa_model, org_data, image_data, programs, action_vector, resnet=None, render=True, batch_idx=0,
             split="Rendered"):

        with torch.no_grad():
            predictions_before, _, _ = vqa_model(**image_data)
            predictions_before = [f.item() for f in predictions_before.max(1)[1]]

        programs = translate_str_program(programs)
        object_positions = self.alter_object_positions_on_action(org_data, action_vector)
        org_data['object_positions'] = object_positions
        scene = translate_state(org_data)
        res = self.oracle(programs, scene, None)
        res = translate_answer(res)
        validity = res
        state_after = org_data

        # TODO: This is from the rendered stuff
        if render:
            rendered_images, questions = self.state2img(org_data, delete_every=False,
                                                        custom_index=batch_idx * self.batch_size, split=split)
            if not hasattr(self, 'cached_rendered_images'):
                self.cached_rendered_images = {}
                self.cached_rendered_images.update({split: []})
                self.cached_questions = {}
                self.cached_questions.update({split: []})
            if split not in self.cached_rendered_images.keys():
                self.cached_rendered_images.update({split: []})
                self.cached_questions.update({split: []})
            self.cached_rendered_images[split].append(rendered_images)
            self.cached_questions[split].append(questions)
        else:
            rendered_images = self.cached_rendered_images[split][batch_idx]
            questions = self.cached_questions[split][batch_idx]
        scene = [f for f in scene]

        predictions_after, softmax_predictions_after = self.perpare_and_pass(vqa_model, resnet, questions,
                                                                             rendered_images, id_list=rendered_images,
                                                                             split=split)

        return predictions_after, softmax_predictions_after, predictions_before, validity, scene, state_after

    def get_rewards(self, vqa_model, org_data, image_data, programs, y_real, action_vector, resnet=None, render=True,
                    batch_idx=0, split='Rendered'):
        predictions_after, softmax_predictions_after, predictions_before, validity, scene, state_after = self.step(
            vqa_model=vqa_model,
            org_data=org_data,
            image_data=image_data,
            programs=programs,
            action_vector=action_vector,
            resnet=resnet, render=render, batch_idx=batch_idx, split=split)
        pb = []
        pa = []
        not_rendered = []
        for b, a in list(itertools.zip_longest(predictions_before, predictions_after)):
            if b is not None:
                if a == -1:
                    pb.append(int(b))
                    pa.append(int(b))
                    not_rendered.append(1)
                else:
                    pb.append(int(b))
                    pa.append(int(a))
                    not_rendered.append(0)
            else:
                break
        predictions_before, predictions_after = pb, pa
        not_rendered = torch.LongTensor(not_rendered)

        if isinstance(predictions_before, list):
            predictions_before = torch.LongTensor(predictions_before)

        if isinstance(predictions_after, list):
            predictions_after = torch.LongTensor(predictions_after)
        if self.batch_size == 1:
            validity = [validity]
        answer_stayed_the_same = y_real - torch.stack(validity, dim=0).long()
        answer_stayed_the_same = 1.0 * answer_stayed_the_same.eq(0).squeeze(1)
        model_answered_correctly = y_real.squeeze(1) - predictions_before
        model_answered_correctly = 1.0 * model_answered_correctly.eq(0)

        confusion_rewards = (model_answered_correctly * answer_stayed_the_same * (
                1.0 * (predictions_before != predictions_after)))

        change_rewards = (answer_stayed_the_same * (1.0 * (predictions_before != predictions_after)))

        fail_rewards = self.fail_weight * torch.ones_like(change_rewards)
        invalid_scene_rewards = self.invalid_weight * not_rendered
        self.rewards = self.confusion_weight * confusion_rewards.numpy() + self.change_weight * change_rewards.numpy() + fail_rewards.numpy() + invalid_scene_rewards.numpy()
        return self.rewards, confusion_rewards, change_rewards, softmax_predictions_after

    @staticmethod
    def quantize(action, effect_range=(-0.3, 0.3), steps=6):
        action = action.detach().cpu().numpy()
        bs_, length_ = action.shape
        quantized_actions = np.empty(action.shape)
        for i in range(bs_):
            for j in range(length_):
                quantized_actions[i, j] = effect_range[0] + action[i, j] * ((effect_range[1] - effect_range[0]) / steps)
        return quantized_actions

    @staticmethod
    def set_model_trainable(model, trainable, but_list=None):
        if trainable:
            if but_list is None:
                model.train()
            else:
                for name, param in model.named_parameters():
                    for entry in but_list:
                        if entry in name:
                            param.requires_grad = True
                            break
                        else:
                            param.requires_grad = False
        else:
            model.eval()
        return

    def clone_adversarial_agent(self, random_weights=False):
        if random_weights:
            self.adversarial_agent_2 = PolicyNet(input_size=128, hidden_size=256, dropout=0.0, reverse_input=True)
        else:
            # Bring to CPU #
            self.adversarial_agent_1.to('cpu')
            self.adversarial_agent_2 = copy.deepcopy(self.adversarial_agent_1)
            # Get both on system device #
            self.adversarial_agent_1.to(self.device)
            self.adversarial_agent_2.to(self.device)

    def interpolation_pipeline(self):
        return

    def engage(self, minigame, minigame2, vqa_model=None, adversarial_agent=None, adversarial_agent_eval=None,
               train_vqa=False,
               train_agent=False):
        if vqa_model is None:
            _print("[Warning] VQA Model in engage is None, assuming self.vqa_model.\n")
            vqa_model = self.vqa_model
            self.set_model_trainable(vqa_model, train_vqa, but_list=['fc2','fc3'])

        if adversarial_agent is None:
            _print("[Warning] Adversarial Agent is None, assuming self.adversarial_agent_1.\n")
            if self.adversarial_agent_1 is None:
                _print(
                    "[Pipeline Error] Adversarial Agent 1 is empty... Make sure "
                    "to load him or pass it as an argument\n Exiting...")
                raise ValueError
            adversarial_agent = self.adversarial_agent_1
            self.set_model_trainable(adversarial_agent, train_agent)

        if adversarial_agent_eval is None:
            _print("[Warning] Adversarial Agent EVAL is None, assuming self.adversarial_agent_2.\n")
            if self.adversarial_agent_2 is None:
                _print(
                    "[Pipeline Error] Adversarial Agent 2 is empty... Make sure "
                    "to load him or pass it as an argument\n Exiting...")
                raise ValueError
            adversarial_agent_eval = self.adversarial_agent_2
            self.set_model_trainable(adversarial_agent_eval, False)

        if self.pipeline == 'extrapolation':
            self.extrapolation_pipeline(minigame=minigame, minigame2=minigame2, vqa_model=vqa_model,
                                        adversarial_agent=adversarial_agent,
                                        adversarial_agent_eval=adversarial_agent_eval,
                                        logger=None)
        else:
            raise NotImplementedError
        return

    def extrapolation_pipeline(self, minigame, minigame2, vqa_model, adversarial_agent, adversarial_agent_eval,
                               logger=None,
                               vqa_model_lr=1e-4,
                               vqa_model_optim='AdamW'):
        """
            In this Scenario, The VQA Agent is Trainable, and we have 2 different adversarial agents that are NOT TRAINABLE
            We fine-tune the VQA Agent on one ADVERSARIAL AGENT and test performance on 2nd.
            AdamW 1e-4 Solved 86.57
            AdamW 5e-4 Solved 86.63
            SGD_GC 1e-4 / 0.9 Solved 66.0
            AdamW 1e-5 Solved 86.63
            SGD_GC 1e-5 / 0.9 Solved 78.5
            AdamW 8e-6 Solved 86.63
            AdamW 5e-6 Not Solved 86.85
            AdamW 1e-6 Not Solved 87.04
            SGD_GC 1e-6 / 0.75 Solved 85.1
            Org 87.05
        """
        RENDER_FLAG = True
        criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction='sum')
        if vqa_model_optim == 'AdamW':
            vqa_model_optimizer = optim.AdamW(
                [{'params': vqa_model.parameters(), 'lr': vqa_model_lr,
                  'weight_decay': 1e-4}
                 ])
        elif vqa_model_optim == 'SGD':
            vqa_model_optimizer = optim.SGD(vqa_model.parameters(), lr=vqa_model_lr, momentum=0.9, weight_decay=1e-4)
        elif vqa_model_optim == 'SGD_GC':
            vqa_model_optimizer = SGD_GC(params=vqa_model.parameters(), lr=vqa_model_lr, momentum=0.9, weight_decay=1e-4)
        else:
            raise ValueError('Optimizers Supported are [AdamW / SGD / SGD_GC]')

        minigame_iter = iter(minigame)
        accuracy_drop = []
        confusion_drop = []
        current_batch_idx = 0
        epochs_passed = 0
        epoch_accuracy_drop = 0
        epoch_confusion_drop = 0
        epoch_accuracy_drop_history = []
        epoch_confusion_drop_history = []

        while epochs_passed < self.defense_rounds:
            try:
                features, org_data, image_data, programs, y_real = self.extract_features(minigame_iter)
            except StopIteration:
                del minigame_iter
                minigame_iter = iter(minigame)
                features, org_data, image_data, programs, y_real = self.extract_features(minigame_iter)

                best_epoch_accuracy_drop = epoch_accuracy_drop / len(minigame)
                best_epoch_confusion_drop = epoch_confusion_drop / len(minigame)
                _print(
                    f"Defense Game Extrapolation | Epoch {epochs_passed} | Epoch Accuracy"
                    f" Drop: {best_epoch_accuracy_drop}% | Epoch Confusion {best_epoch_confusion_drop} % ")
                if logger is not None:
                    logger.log({'Epoch': epochs_passed,
                                'Drop': best_epoch_accuracy_drop,
                                'Consistency': best_epoch_confusion_drop})

                epoch_accuracy_drop_history.append(epoch_accuracy_drop / len(minigame))
                epoch_confusion_drop_history.append(epoch_confusion_drop / len(minigame))
                if best_epoch_accuracy_drop == 0:
                    break
                epoch_accuracy_drop = 0
                epoch_confusion_drop = 0
                current_batch_idx = 0
                # Here we should test on other game / other agent #
                if epochs_passed % 5 == 0:
                    self.evaluate(epoch=epochs_passed, minigame=minigame2, vqa_model=vqa_model,
                                 adversarial_agent_eval=adversarial_agent_eval, render_flag=RENDER_FLAG)
                RENDER_FLAG = False
                epochs_passed += 1

            # Make Agent 2 Suggest Changes on Batch #
            sx, sy, _ = adversarial_agent(features, None)
            actionsx = torch.cat([f.unsqueeze(1) for f in sx], dim=1).max(2)[1]
            actionsy = torch.cat([f.unsqueeze(1) for f in sy], dim=1).max(2)[1]
            action = torch.cat([actionsx, actionsy], dim=1)
            # Quantize Changes #
            mixed_actions = self.quantize(action)
            # Now is the important part, we need:
            # - VQA Predictions Before in List Format
            # - VQA Predictions After in List Format
            # - VQA Predictions After in Softmax Format

            _, confusion_rewards, change_rewards, vqa_predictions_softmax = self.get_rewards(
                vqa_model=vqa_model, org_data=org_data, image_data=image_data, programs=programs, y_real=y_real,
                action_vector=mixed_actions,
                resnet=self.resnet, render=RENDER_FLAG, batch_idx=current_batch_idx)
            # In order not to overfit in this region, we will use the rewards as a mask of what to backprop #
            # We need to backprop only the images that are incorrectly classified after the agent not the rest #
            mask = torch.FloatTensor(confusion_rewards).to(self.device)
            if vqa_predictions_softmax is not None:
                y_real = y_real.to('cuda')
                loss = 0
                for i in range(self.batch_size):
                    loss = loss + criterion(vqa_predictions_softmax[i, :].unsqueeze(0), y_real[i, :]) * mask[i]

                loss.backward()

                torch.nn.utils.clip_grad_norm_(vqa_model.parameters(), 50)
                vqa_model_optimizer.step()
                vqa_model_optimizer.zero_grad()
            else:
                _print("Nothing was Rendered!")

            batch_accuracy = 100 * (confusion_rewards.mean()).item()
            batch_confusion = 100 * (change_rewards.mean()).item()
            accuracy_drop.append(batch_accuracy)
            confusion_drop.append(batch_confusion)
            epoch_accuracy_drop += batch_accuracy
            epoch_confusion_drop += batch_confusion
            current_batch_idx += 1

        if logger is not None:
            logger.log({'Epoch': epochs_passed, 'Drop': max(epoch_accuracy_drop_history),
                        'Consistency': max(epoch_confusion_drop_history)})
        self.save(model=vqa_model, range=self.train_range,
                  path=f'./results/experiment_defense/visual/{self.train_name}')
        self.evaluate(epoch=epochs_passed, minigame=minigame2, vqa_model=vqa_model,
                      adversarial_agent_eval=adversarial_agent_eval, render_flag=RENDER_FLAG)
        return max(epoch_accuracy_drop_history), max(epoch_confusion_drop_history)

    def evaluate(self, epoch, minigame, vqa_model, adversarial_agent_eval, render_flag=True):
        current_batch_idx = 0
        minigame_iter = iter(minigame)
        accuracy_drop = []
        confusion_drop = []
        epoch_accuracy_drop = 0
        epoch_confusion_drop = 0

        while True:
            try:
                features, org_data, image_data, programs, y_real = self.extract_features(minigame_iter)
            except StopIteration:
                del minigame_iter

                best_epoch_accuracy_drop = epoch_accuracy_drop / len(minigame)
                best_epoch_confusion_drop = epoch_confusion_drop / len(minigame)
                _print(
                    f"Defense Game Extrapolation Evaluation Round | Epoch {epoch} | Epoch Accuracy"
                    f" Drop: {best_epoch_accuracy_drop}% | Epoch Confusion {best_epoch_confusion_drop} % ")
                return best_epoch_accuracy_drop, best_epoch_confusion_drop

            sx, sy, _ = adversarial_agent_eval(features, None)
            actionsx = torch.cat([f.unsqueeze(1) for f in sx], dim=1).max(2)[1]
            actionsy = torch.cat([f.unsqueeze(1) for f in sy], dim=1).max(2)[1]
            action = torch.cat([actionsx, actionsy], dim=1)
            mixed_actions = self.quantize(action)
            _, confusion_rewards, change_rewards, vqa_predictions_softmax = self.get_rewards(
                vqa_model=vqa_model, org_data=org_data, image_data=image_data, programs=programs, y_real=y_real,
                action_vector=mixed_actions,
                resnet=self.resnet, render=render_flag, batch_idx=current_batch_idx, split='Evaluation')

            batch_accuracy = 100 * (confusion_rewards.mean()).item()
            batch_confusion = 100 * (change_rewards.mean()).item()
            accuracy_drop.append(batch_accuracy)
            confusion_drop.append(batch_confusion)
            epoch_accuracy_drop += batch_accuracy
            epoch_confusion_drop += batch_confusion
            current_batch_idx += 1

    def assess_overall_drop(self, train_name, train_range):
        if train_name is None:
            _print('Train Name is Empty... Loading self.train_name of Defense Game\n')
            train_name = self.train_name

        if train_range is None:
            _print('Train Range is Empty... Loading self.train_range of Defense Game\n')
            train_range = self.train_range

        path = f'./results/experiment_defense/visual/{train_name}/defense_against_{train_range}.pt'
        if train_name == 'rnfp':
            model = load_rnfp(path)
            loader = load_loader_rnfp()
            results = inference_with_rnfp(loader=loader, model=model, resnet_extractor=None, evaluator=True)

        else:
            raise NotImplementedError(f'Assess overall drop expects input rnfp, you entered: {train_name}')


        return results
