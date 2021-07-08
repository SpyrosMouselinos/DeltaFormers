import json
import os.path as osp
import sys

import tqdm
import yaml

sys.path.insert(0, osp.abspath('.'))

import argparse
from modules.embedder import *
from utils.train_utils import StateCLEVR, ImageCLEVR

AVAILABLE_DATASETS = {
    'DeltaRN': StateCLEVR,
    'DeltaSQFormer': StateCLEVR,
    'DeltaQFormer': StateCLEVR,
    'DeltaRNFP': ImageCLEVR,
}

AVAILABLE_MODELS = {'DeltaRN': DeltaRN,
                    'DeltaRNFP': DeltaRNFP,
                    'DeltaSQFormer': DeltaSQFormer,
                    'DeltaQFormer': DeltaQFormer}


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


def kwarg_dict_to_device(data_obj, device):
    if device == 'cpu':
        return data_obj
    cpy = {}
    for key, _ in data_obj.items():
        cpy[key] = data_obj[key].to(device)
    return cpy


def load_encoders():
    with open('../translation_tables.yaml', 'r') as fin:
        translation = yaml.load(fin, Loader=yaml.FullLoader)['translation']
    with open(f'../data/vocab.json', 'r') as fin:
        parsed_json = json.load(fin)
        q2index = parsed_json['question_token_to_idx']
        a2index = parsed_json['answer_token_to_idx']

    return translation, q2index, a2index


def load_model(device, load_from=None):
    if device == 'cuda':
        device = 'cuda:0'
    experiment_name = load_from.split('results/')[-1].split('/')[0]
    config = f'../results/{experiment_name}/config.yaml'
    with open(config, 'r') as fin:
        config = yaml.load(fin, Loader=yaml.FullLoader)

    model = AVAILABLE_MODELS[config['model_architecture']](config)
    model = model.to(device)
    checkpoint = torch.load(load_from)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loading Model of type: {config['model_architecture']}\n")
    model.eval()
    return model


def encode_questions_and_scenes(question, q2index, a2index, scene, translation):
    image_index_scene, n_objects, object_positions, object_colors, object_shapes, object_materials, object_sizes = \
        single_scene_translator(scene=scene, translation=translation)

    image_index_question, n_tokens, q, a = single_question_parser(question,
                                                                  word_replace_dict={'true': 'yes',
                                                                                     'True':'yes',
                                                                                     'false': 'no',
                                                                                     'False':'no'
                                                                                     },
                                                                  q2index=q2index,
                                                                  a2index=a2index)
    if n_tokens is None:
        return 0, 0

    if image_index_scene == image_index_question:
        types = [1] * n_objects + [0] * (10 - n_objects) + [2] * n_tokens + [0] * (50 - n_tokens)
        types = torch.LongTensor(types).view(60)
        positions = torch.LongTensor([0] * 10 + list(range(1, n_tokens + 1)) + [0] * (50 - n_tokens)).view(60)
        x = {'positions': positions.unsqueeze(0),
             'types': types.unsqueeze(0),
             'object_positions': object_positions.unsqueeze(0),
             'object_colors': object_colors.unsqueeze(0),
             'object_shapes': object_shapes.unsqueeze(0),
             'object_materials': object_materials.unsqueeze(0),
             'object_sizes': object_sizes.unsqueeze(0),
             'question': q.unsqueeze(0),
             }
        y = a
    else:
        print(f"Image index {image_index_scene} and question index {image_index_question} do not match!\n")
        return 1,1

    return x, y


def fp(model, x, y, device):
    with torch.no_grad():
        x = kwarg_dict_to_device(x, device=device)
        y = y.to(device)
        model = model.to(device)
        y_pred = model(**x)[0]
    acc = (y_pred.argmax(1) == y).float().detach().cpu().numpy()
    return acc


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--device', type=str, help='cpu or cuda', default='cuda')
#     parser.add_argument('--load_from', type=str, help='continue training',
#                         default='../results/experiment_sq/mos_epoch_124.pt')
#     args = parser.parse_args()
    # model = load_model(device=args.device, load_from=args.load_from)
    #
    # translation, q2index, a2index = load_encoders()
    #
    # with open(f'../data/CLEVR_train_scenes.json', 'r') as fin:
    #     parsed_json = json.load(fin)
    #     scenes = parsed_json['scenes']
    #
    # with open(f'../data/CLEVR_train_questions.json', 'r') as fin:
    #     parsed_json = json.load(fin)
    #     questions = parsed_json['questions']
    #
    # acc = 0
    # eligible = 0
    # question_counter = 0
    # scene_counter = 0
    # while scene_counter < 5000:
    #     scene = scenes[scene_counter]
    #     question = questions[question_counter]
    #     x, y = encode_questions_and_scenes(question, q2index, a2index, scene, translation)
    #     if x == 0 and y == 0:
    #         question_counter += 1
    #         continue
    #     elif x == 1 and y == 1:
    #         scene_counter += 1
    #         continue
    #     else:
    #         eligible += 1
    #         acc += fp(model, x, y, device=args.device)[0]
    #         question_counter += 1
    #
    # print(acc / eligible)
