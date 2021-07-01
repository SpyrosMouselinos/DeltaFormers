import json
import os.path as osp
import pickle
import sys

from tqdm import trange

sys.path.insert(0, osp.abspath('..'))
import torch
import yaml
from torch.utils.data import Dataset


def scene_parser(mode='val'):
    with open(f'data/CLEVR_{mode}_scenes.json', 'r') as fin:
        parsed_json = json.load(fin)
        scenes = parsed_json['scenes']
    return scenes


def question_parser(mode='val'):
    with open(f'data/CLEVR_{mode}_questions.json', 'r') as fin:
        parsed_json = json.load(fin)
        questions = parsed_json['questions']
    return questions


def single_scene_translator(scene: dict, translation: dict):
    image_index = scene['image_index']
    n_objects = len(scene['objects'])

    xs = []
    ys = []
    thetas = []
    colors = []
    materials = []
    shapes = []
    sizes = []
    for obj in scene['objects']:
        xs.append(obj['3d_coords'][0] / 3)
        ys.append(obj['3d_coords'][1] / 3)
        thetas.append(obj['3d_coords'][2] / 360)
        colors.append(translation[obj['color']])
        materials.append(translation[obj['material']])
        shapes.append(translation[obj['shape']])
        sizes.append(translation[obj['size']])

    #######################################################
    object_positions_x = torch.FloatTensor(xs + (10 - n_objects) * [0]).unsqueeze(1)
    object_positions_y = torch.FloatTensor(ys + (10 - n_objects) * [0]).unsqueeze(1)
    object_positions_t = torch.FloatTensor(thetas + (10 - n_objects) * [0]).unsqueeze(1)

    object_positions = torch.cat([object_positions_x, object_positions_y, object_positions_t], 1).view(10, 3)
    object_colors = torch.LongTensor(colors + (10 - n_objects) * [0])

    object_shapes = torch.LongTensor(shapes + (10 - n_objects) * [0])
    object_materials = torch.LongTensor(materials + (10 - n_objects) * [0])
    object_sizes = torch.LongTensor(sizes + (10 - n_objects) * [0])

    return image_index, n_objects, object_positions, object_colors, object_shapes, object_materials, object_sizes


def single_question_parser(question: dict, word_replace_dict: dict, q2index: dict, a2index: dict):
    image_index = question['image_index']
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
                return image_index, None, None, None
            elif '?' in word or ';' in word:
                tokenized_q.append(q2index[word[:-1]])
                tokenized_q.append(q2index[';'])
            else:
                tokenized_q.append(q2index[word])
        q = torch.LongTensor(tokenized_q + [0] * (50 - len(tokenized_q))).view(50)
    if a2index is None:
        pass
    else:
        a = torch.LongTensor([a2index[a] - 4])

    return image_index, len(tokenized_q), q, a


def scene_image_matcher(split, translation, q2index, a2index):
    ### All scenes ###
    scenes = scene_parser(split)

    ### All questions ###
    questions = question_parser(split)

    x_samples = []
    y_samples = []
    scene_counter = 0
    question_counter = 0
    for scene_counter in trange(len(scenes)):
        image_index_scene, n_objects, object_positions, object_colors, object_shapes, object_materials, object_sizes = \
            single_scene_translator(scene=scenes[scene_counter], translation=translation)
        while question_counter < len(questions):
            image_index_question, n_tokens, q, a = single_question_parser(questions[question_counter],
                                                                          word_replace_dict={'True': 'yes',
                                                                                             'False': 'no'},
                                                                          q2index=q2index,
                                                                          a2index=a2index)
            # Bad question Move on #
            if q is None and a is None:
                question_counter += 1
                continue

            if image_index_scene == image_index_question:
                types = [1] * n_objects + [0] * (10 - n_objects) + [2] * n_tokens + [0] * (50 - n_tokens)
                types = torch.LongTensor(types).view(60)
                positions = torch.LongTensor([0] * 10 + list(range(1, n_tokens + 1)) + [0] * (50 - n_tokens)).view(60)
                x_samples.append({'positions': positions,
                                  'types': types,
                                  'object_positions': object_positions,
                                  'object_colors': object_colors,
                                  'object_shapes': object_shapes,
                                  'object_materials': object_materials,
                                  'object_sizes': object_sizes,
                                  'question': q,
                                  })
                y_samples.append(a)

                # Increment and Loop #
                question_counter += 1
            else:
                # Question is for the next image #
                break
    return x_samples, y_samples


class StateCLEVR(Dataset):
    """CLEVR dataset made from Scene States."""

    def __init__(self, config=None, split='val'):

        # if config is None:
        #     with open(osp.dirname(osp.dirname(__file__)) + '/config.yaml', 'r') as fin:
        #         config = yaml.load(fin, Loader=yaml.FullLoader)

        if osp.exists(f'data/{split}_dataset.pt'):
            with open(f'data/{split}_dataset.pt', 'rb')as fin:
                info = pickle.load(fin)
            self.split = info['split']
            self.translation = info['translation']
            self.q2index = info['q2index']
            self.a2index = info['a2index']
            self.x = info['x']
            self.y = info['y']
            print("Dataset loaded succesfully!\n")
        else:
            with open(osp.dirname(osp.dirname(__file__)) + '/translation_tables.yaml', 'r') as fin:
                translation = yaml.load(fin, Loader=yaml.FullLoader)['translation']
            with open(f'data/vocab.json', 'r') as fin:
                parsed_json = json.load(fin)
                q2index = parsed_json['question_token_to_idx']
                a2index = parsed_json['answer_token_to_idx']

            self.split = split
            # self.config = config
            self.translation = translation
            self.q2index = q2index
            self.a2index = a2index
            x, y = scene_image_matcher(self.split, self.translation, self.q2index, self.a2index)
            self.x = x
            self.y = y
            print("Dataset loaded succesfully!...Saving\n")
            info = {
                'split': self.split,
                'translation': self.translation,
                'q2index': self.q2index,
                'a2index': self.a2index,
                'x': self.x,
                'y': self.y
            }
            with open(f'data/{self.split}_dataset.pt', 'wb') as fout:
                pickle.dump(info, fout)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.x[idx], self.y[idx]

# import matplotlib.pyplot as plt
# with open('../config.yaml', 'r') as fin:
#     config = yaml.load(fin, Loader=yaml.FullLoader)
#
# with open('../translation_tables.yaml', 'r') as fin:
#     trans = yaml.load(fin, Loader=yaml.FullLoader)
# test_loader = torch.utils.data.DataLoader(StateCLEVR(config=config, split='val'), batch_size=1, shuffle=False)
#
# with open(f'../data/vocab.json', 'r') as fin:
#     parsed_json = json.load(fin)
#     q2index = parsed_json['question_token_to_idx']
#     a2index = parsed_json['answer_token_to_idx']
#
# index2q = {v: k for k, v in q2index.items()}
# index2a = {v: k for k, v in a2index.items()}
#
# for index, (x, y) in enumerate(test_loader):
#     positions = x['positions'][0].numpy()
#     types = x['types'][0].numpy()
#     object_positions = x['object_positions'][0].numpy()
#     object_colors = [trans['reverse_translation_color'][f] if f != 0 else None for f in x['object_colors'][0].numpy()]
#     object_shapes = [trans['reverse_translation_shape'][f] if f != 0 else None for f in x['object_shapes'][0].numpy()]
#     object_materials = [trans['reverse_translation_material'][f] if f != 0 else None for f in
#                         x['object_materials'][0].numpy()]
#     object_sizes = [trans['reverse_translation_size'][f] if f != 0 else None for f in x['object_sizes'][0].numpy()]
#     q = [index2q[f] for f in x['question'][0].numpy()]
#     a = index2a[y.item()]
#
#     mmt = {
#         'cube': 's',
#         'cylinder': 'h',
#         'sphere': 'o',
#     }
#
#     mst = {
#         'large': 8,
#         'small': 6
#     }
#     plt.figure(figsize=(10, 10))
#     plt.title(f"Bird Eye View of image: {index}")
#     #plt.rcParams['axes.facecolor'] = 'black'
#     for oi in range(0, 10):
#         x = object_positions[oi][0]
#         y = object_positions[oi][1]
#         if x != 0 and y != 0:
#             plt.scatter(x=x, y=y, c=object_colors[oi], s=mst[object_sizes[oi]]**2, marker=mmt[object_shapes[oi]])
#     print(f"Question: {' '.join(q)}")
#     print(f"Answer: {a}")
#     plt.xlim(-1,1)
#     plt.ylim(-1,1)
#     plt.show()
#     if index == 20:
#         break
