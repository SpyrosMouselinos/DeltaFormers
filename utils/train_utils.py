import sys
import os.path as osp
sys.path.insert(0, osp.abspath('.'))

import yaml
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def scene_parser(mode='val'):
    with open(f'data/CLEVR_{mode}_scenes.json','r') as fin:
        parsed_json = json.load(fin)
        scenes = parsed_json['scenes']
    return scenes

def question_parser(mode='val'):
    with open(f'data/CLEVR_{mode}_questions.json','r') as fin:
        parsed_json = json.load(fin)
        questions = parsed_json['questions']
    return questions

def single_scene_translator(scene: dict, translation: dict):
    image_index = scene['image_index']
    n_objects = len(scene['objects'])
    key = 1.0
    back = 1.0
    fill = 1.0
    camera = 0.5
    xs = []
    ys = []
    thetas = []
    colors =[]
    materials = []
    shapes = []
    sizes = []
    for obj in scene['objects']:
        xs.append(obj['3d_coords'][0] / 3.0) 
        ys.append(obj['3d_coords'][1] / 3.0) 
        thetas.append(obj['3d_coords'][2] / 360) 
        colors.append(translation[obj['color']])
        materials.append(translation[obj['material']])
        shapes.append(translation[obj['shape']])
        sizes.append(translation[obj['size']])

    
    #######################################################
    positions = torch.LongTensor(list(range(0, 61))).view(1,61)
    types = [1] * n_objects + [0] * (10 - n_objects) + [3] + [2] * 50
    types = torch.LongTensor(types).view(1,61)
    object_positions_x = torch.FloatTensor(xs + (10 - n_objects) * [0]).unsqueeze(1)
    object_positions_y = torch.FloatTensor(ys + (10 - n_objects) * [0]).unsqueeze(1)
    object_positions_t = torch.FloatTensor(thetas + (10 - n_objects) * [0]).unsqueeze(1)
    
    object_positions = torch.cat([object_positions_x,object_positions_y,object_positions_t], 1).view(1, 10, 3)
    object_colors    = torch.LongTensor(colors + (10 - n_objects) * [0]).unsqueeze(0)
    
    object_shapes    = torch.LongTensor(shapes + (10 - n_objects) * [0]).unsqueeze(0)
    object_materials = torch.LongTensor(materials + (10 - n_objects) * [0]).unsqueeze(0)
    object_sizes     = torch.LongTensor(sizes + (10 - n_objects) * [0]).unsqueeze(0)
    
    scene_state      = torch.FloatTensor([key, back, fill, camera]).unsqueeze(0).unsqueeze(1)

    return positions, types, object_positions, object_colors, object_shapes, object_materials, object_sizes, scene_state

def single_question_parser(question:dict, word_replace_dict:dict, q2index:dict, a2index:dict):
    image_id = question['image_filename']
    q = str(question['question'])
    a = str(question['answer'])
    if word_replace_dict is None:
        pass
    else:
        for word, replacement in word_replace_dict.items():
            q = q.replace(word, replacement)
            a = a.replace(word, replacement)
    if q2index is None:
        pass
    else:
        q = '<START>' + ' ' + q + ' ' + '<END>'
        tokenized_q = []
        for word in q.split(' '):
            if 'bullet' in word or 'butterfly' in word:
                return None, None
            elif '?' in word or ';' in word:
                tokenized_q.append(q2index[word[:-1]])
                tokenized_q.append(q2index[';'])
            else:
                tokenized_q.append(q2index[word])
        q = torch.LongTensor(tokenized_q + [0] * (50 - len(tokenized_q))).view(1,50)
    if a2index is None:
            pass
    else:
        a = torch.LongTensor([a2index[a]])
    return q, a

class StateCLEVR(Dataset):
    """CLEVR dataset made from Scene States."""

    def __init__(self, config=None, split='val'):
        if config is None:
            with open(osp.dirname(osp.dirname(__file__)) + '/config.yaml', 'r') as fin:
                config =yaml.load(fin, Loader=yaml.FullLoader)
        with open(osp.dirname(osp.dirname(__file__)) + '/translation_tables.yaml', 'r') as fin:
            translation = yaml.load(fin, Loader=yaml.FullLoader)['translation']
        with open(f'data/vocab.json','r') as fin:
            parsed_json = json.load(fin)
            q2index = parsed_json['question_token_to_idx']
            a2index = parsed_json['answer_token_to_idx']
            
        self.split = split
        self.config = config
        self.translation = translation
        self.q2index = q2index
        self.a2index = a2index
        self.scenes = scene_parser(split)
        self.questions = question_parser(split)

    def __len__(self):
        return self.config['num_questions_per_image'] * len(self.scenes)

    def __getitem__(self, idx):
        # q = q.to(device)
        # a = a.to(device)
        # positions = positions.to(device)
        # types = types.to(device)
        # object_positions = object_positions.to(device)
        # object_colors = object_colors.to(device)
        # object_shapes = object_shapes.to(device)
        # object_materials = object_materials.to(device)
        # object_sizes = object_sizes.to(device)
        # scene_state = scene_state.to(device)
        if torch.is_tensor(idx):
            idx = idx.tolist()

        positions, types, object_positions, object_colors, object_shapes, object_materials, object_sizes, scene_state = \
            single_scene_translator(scene=self.scenes[idx // self.config['num_questions_per_image']], translation=self.translation)

        q, a = single_question_parser(self.questions[idx], word_replace_dict={'True':'yes','False':'no'}, q2index=self.q2index, a2index=self.a2index)
        if q is None:
            return self.__getitem__(idx-1)
        sample = {'positions': positions,
                  'types': types,
                  'object_positions':object_positions,
                  'object_colors':object_colors,
                  'object_shapes':object_shapes,
                  'object_materials':object_materials,
                  'object_sizes':object_sizes,
                  'scene_state':scene_state,
                  'question':q,
                  'answer':a
                  }
        return sample
    
