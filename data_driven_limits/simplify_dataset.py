import copy
import json
import os
import os.path as osp
import shutil
import sys

import matplotlib.pyplot as plt
from natsort import natsorted
from skimage.io import imread, imshow

sys.path.insert(0, osp.abspath('.'))
from utils.train_utils import StateCLEVR, ImageCLEVR_HDF5, MixCLEVR_HDF5
import argparse
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

DEFAULT_PATH = osp.dirname(osp.abspath(__file__)).replace('data_driven_limits', 'neural_render')
DEFAULT_DATASET_LOCATION = osp.dirname(osp.abspath(__file__)).replace('_driven_limits', '')


def interleave_list(ql, skip_limit=1782, number_of_limits=8):
    new_l = []
    for i in range(0, skip_limit):
        for j in range(0, number_of_limits):
            new_l.append(ql[i + j * skip_limit])
    return new_l


def sanitize(qoa, dicta):
    tokenized = []
    for token in qoa.split(' '):
        if token in dicta:
            tokenized.append(dicta[token])
        else:
            print("Word {} does not exist in dict, using ID 3")
            tokenized.append(3)
    return tokenized


def show_image(dataset_name, index):
    img = imread(
        DEFAULT_DATASET_LOCATION + f'/images/{dataset_name}/' + f'CLEVR_{dataset_name}_{add_nulls2(index, 6)}.png')
    plt.figure()
    plt.title(f"View of dataset {dataset_name} / image id {index}")
    imshow(img)
    plt.show()
    plt.close()
    return


def load_legit_words():
    with open(f'{DEFAULT_DATASET_LOCATION}/vocab.json', 'r') as fin:
        data = json.load(fin)
        qtoken2index = data['question_token_to_idx']
        qindex2token = {v: k for k, v in qtoken2index.item()}
        atoken2index = data['answer_token_to_idx']
        aindex2token = {v: k for k, v in atoken2index.item()}
        return qtoken2index, qindex2token, atoken2index, aindex2token


def add_nulls2(int, cnt):
    nulls = str(int)
    for i in range(cnt - len(str(int))):
        nulls = '0' + nulls
    return nulls


def detect_dataset(templates, kind='images', path=DEFAULT_PATH):
    mathced_counts = [0] * len(templates)
    matched_files = []
    image_path = path + f'\{kind}'
    image_files = os.listdir(image_path)
    for i, template in enumerate(templates):
        for file in image_files:
            if template in file:
                matched_files.append(file)
                mathced_counts[i] += 1
    print(f"Found {sum(mathced_counts)} total {kind} files.\n")
    for i, template in enumerate(templates):
        print(f'Found {mathced_counts[i]} {kind} files in template {template}\n')
    return natsorted(matched_files), mathced_counts


def merge_scenes(list_of_scene_files, dataset_name):
    new_scenes = {
        "info": {"split": f"{dataset_name}", "license": "Creative Commons Attribution (CC BY 4.0)", "version": "1.0",
                 "date": "2/11/2021"}, "scenes": []
    }
    for i, scene in enumerate(list_of_scene_files):
        with open(scene, 'r') as fin:
            data = json.load(fin)
            new_data = copy.deepcopy(data)
            new_data['split'] = dataset_name
            new_data['image_index'] = i
            new_data['image_filename'] = f'CLEVR_{dataset_name}_{add_nulls2(i, 6)}.png'
            new_scenes['scenes'].append(new_data)

    with open(f'{DEFAULT_DATASET_LOCATION}/CLEVR_{dataset_name}_scenes.json', 'w') as fout:
        json.dump(new_scenes, fout)

    return


def copy_images_and_scenes(dataset_name, images=None, scenes=None):
    if osp.exists(DEFAULT_DATASET_LOCATION + f'/images/{dataset_name}'):
        pass
    else:
        os.mkdir(DEFAULT_DATASET_LOCATION + f'/images/{dataset_name}')
    if images is None:
        pass
    else:
        for file in images:
            shutil.copy(DEFAULT_PATH + '/images/' + file,
                        DEFAULT_DATASET_LOCATION + f'/images/{dataset_name}/' + file)

    if scenes is None:
        pass
    else:
        merge_scenes(list_of_scene_files=[DEFAULT_PATH + '/scenes/' + f for f in scenes], dataset_name=dataset_name)
    return


def simplify_images(dataset_name, matched_image_files):
    try:
        for i, image in enumerate(natsorted(
                [DEFAULT_DATASET_LOCATION + f'/images/{dataset_name}/' + file for file in matched_image_files])):
            os.rename(image,
                      DEFAULT_DATASET_LOCATION + f'/images/{dataset_name}/' + f'CLEVR_{dataset_name}_{add_nulls2(i, 6)}.png')
    except FileNotFoundError:
        pass
    return


def make_questions(dataset_name, matched_image_counts, interactive=False):
    new_questions = {
        'info': {'split': f'{dataset_name}',
                 'license': 'Creative Commons Attribution (CC BY 4.0)',
                 'version': '1.0',
                 'date': '2/11/2021'},
        'questions': []
    }
    question_template = {'image_index': -1,
                         'program': [],
                         'question_index': 0,
                         'image_filename': None,
                         'question_family_index': 0,
                         'split': f'{dataset_name}',
                         'answer': 'Blah',
                         'question': 'Blah'}

    print(f"Remember you have {sum(matched_image_counts)} total images split on {matched_image_counts}\n")
    questions_added = 0
    add_more = True
    while add_more:
        default_index = 0
        for i in range(0, len(matched_image_counts)):
            flag = True
            print(f"Swap {i}...See a characteristic Image...\n")
            show_image(dataset_name=dataset_name, index=default_index)

            if interactive:
                while flag:
                    print("Enter question for this swap...\n")
                    question = input().strip()
                    print("Enter answer for this question...\n")
                    answer = input().strip().lower()
                    print("Confirm entry?\n")
                    if input().strip().lower() == 'y':
                        flag = False
                    else:
                        flag = True
                print(f"Adding the question to the {i} swap\n, Total: {questions_added}")
                for j in range(default_index, default_index + matched_image_counts[i]):
                    qt = copy.deepcopy(question_template)
                    qt['image_index'] = j
                    qt['image_filename'] = f'CLEVR_{dataset_name}_{add_nulls2(j, 6)}'
                    qt['question'] = question
                    qt['answer'] = answer
                    new_questions['questions'].append(qt)
            else:
                pass
            default_index += matched_image_counts[i]
        print("Want to add more questions?\n")
        response = input().strip().lower()
        if response == 'y':
            add_more = True
            questions_added += 1
        else:
            add_more = False
            questions_added += 1

    new_questions['questions'] = interleave_list(new_questions['questions'], skip_limit=sum(matched_image_counts),
                                                 number_of_limits=questions_added)
    # Wrap Up
    with open(f'{DEFAULT_DATASET_LOCATION}/CLEVR_{dataset_name}_questions.json', 'w') as fout:
        json.dump(new_questions, fout)
        print("QA saved succesfully!")
    return


def prepare_into_dataloaders(dataset_name):
    StateCLEVR(config=None,
               split=dataset_name,
               scenes_path=DEFAULT_DATASET_LOCATION,
               questions_path=DEFAULT_DATASET_LOCATION,
               clvr_path=DEFAULT_DATASET_LOCATION,
               use_cache=False,
               return_program=False,
               effective_range=None,
               effective_range_offset=0)

    ImageCLEVR_HDF5(config=None,
                    split=dataset_name,
                    clvr_path=DEFAULT_DATASET_LOCATION,
                    questions_path=DEFAULT_DATASET_LOCATION,
                    scenes_path=DEFAULT_DATASET_LOCATION,
                    use_cache=False,
                    return_program=False,
                    effective_range=None,
                    output_shape=224,
                    effective_range_offset=0)

    MixCLEVR_HDF5(config=None,
                  split=dataset_name,
                  clvr_path=DEFAULT_DATASET_LOCATION,
                  questions_path=DEFAULT_DATASET_LOCATION,
                  scenes_path=DEFAULT_DATASET_LOCATION,
                  use_cache=False,
                  return_program=False,
                  effective_range=None,
                  output_shape=224,
                  effective_range_offset=0)
    return


def main(dataset_name, dataset_templates, interactive):
    matched_image_files, matched_image_counts = detect_dataset(templates=dataset_templates, kind='images')
    matched_scene_files, matched_scene_counts = detect_dataset(templates=dataset_templates, kind='scenes')
    if matched_image_counts != matched_scene_counts:
        print(f"Same Template Junk Exist! Found {matched_scene_counts} scenes and {matched_image_counts} images!\n")
    copy_images_and_scenes(dataset_name=dataset_name, images=matched_image_files, scenes=matched_scene_files)
    simplify_images(dataset_name, matched_image_files)
    # make_questions(dataset_name, matched_scene_counts, interactive=interactive)
    # prepare_into_dataloaders(dataset_name)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--noo', type=int, default=3)
    parser.add_argument('--nos', type=int, default=0)
    parser.add_argument('--name', type=str, default='Grid_Test')
    parser.add_argument('--interactive', type=str, default='False')
    args = parser.parse_args()

    dataset_templates = [f'CLEVR_{args.name}_{args.noo}it_{f}' for f in range(0, args.nos + 1)]

    main(dataset_name=args.name, dataset_templates=dataset_templates, interactive=eval(args.interactive))
