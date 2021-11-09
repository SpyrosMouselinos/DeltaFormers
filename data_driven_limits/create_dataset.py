import os
import os.path as osp
import sys

sys.path.insert(0, osp.abspath('.'))

import argparse
import random
import warnings
import torch
import shutil
import numpy as np
import pickle
from tqdm import tqdm
from neural_render.blender_render_utils.constants import OUTPUT_IMAGE_DIR_, OUTPUT_SCENE_DIR_, \
    OUTPUT_SCENE_FILE_, \
    OUTPUT_QUESTION_FILE_, UP_TO_HERE_
from neural_render.blender_render_utils.helpers import initialize_paths
from data_driven_limits.utilities import *

warnings.filterwarnings("ignore", category=UserWarning)
random.seed(666)
np.random.seed(666)


def paragontiko(num):
    prod = 1
    for i in range(1, num + 1):
        prod = prod * i
    return prod


def initialize_scene_seed(scenes_path, scene_id):
    print(f'{scenes_path}/val_dataset.pt')
    if osp.exists(f'{scenes_path}/val_dataset.pt'):
        with open(f'{scenes_path}/val_dataset.pt', 'rb') as fin:
            info = pickle.load(fin)
            x = info['x'][scene_id]
    return x


def generate_dataset(number_of_objects, number_of_swaps, image_seed, agent_percentage_start=0, agent_percentage_end=10,
                     random_agent_choice=True,
                     split_name='Limit'):
    if number_of_objects < 2 or number_of_objects > 3:
        raise NotImplementedError("Currently Supporting 2 or 3 items")
    if number_of_swaps < 0 or number_of_swaps > paragontiko(number_of_objects):
        raise NotImplementedError("Currently Swaps between 0 and #items!")
    wizard = Wizard(number_of_objects)
    agent_percentage_diff = agent_percentage_end - agent_percentage_start
    if agent_percentage_diff <= 0:
        raise ValueError(f"You entered start: {agent_percentage_start} / end: {agent_percentage_end}!")
    ppi = int(agent_percentage_diff / 100 * len(wizard.action_memory))
    ppi_start = int(agent_percentage_start / 100 * len(wizard.action_memory))
    pbar = tqdm(total=ppi)
    created = 0
    missed = 0

    if random_agent_choice:
        items = random.sample(range(0, len(wizard.action_memory) - 1), ppi)
    else:
        items = np.arange(ppi_start, ppi_start+ppi)

    for i in items:
        actionsx, actionsy = wizard.act(i)
        actionsx = actionsx - 3
        actionsy = actionsy - 3
        # actionsx = actionsx / 3.0
        # actionsy = actionsy / 3.0
        if number_of_objects == 2:
            actionsx = torch.cat([actionsx, torch.FloatTensor([0])])
            actionsy = torch.cat([actionsy, torch.FloatTensor([0])])
        # Place everything at Center #
        image_seed['object_positions'] *= 0
        result = state2img(state=image_seed, custom_index=i, bypass=False, retry=False, perturbations_x=actionsx,
                           perturbations_y=actionsy, swaps=number_of_swaps, pad=number_of_objects == 2,
                           split_name=split_name)
        if result >= 1:
            created += result
            missed += paragontiko(number_of_objects) - result
        else:
            missed += paragontiko(number_of_objects)
        pbar.update(1)
        pbar.set_postfix({'Created': created, 'Missed': missed})
    pbar.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--noo', type=int, default=3)
    parser.add_argument('--nos', type=int, default=0)
    parser.add_argument('--agentperstart', type=int, default=5)
    parser.add_argument('--agentperend', type=int, default=10)
    parser.add_argument('--randac', type=str, default='False')
    parser.add_argument('--split_name', type=str, default='Grid_Test')
    parser.add_argument('--image_seed_id', type=int, default=60)
    parser.add_argument('--image_seed_path', type=str, default='None')
    parser.add_argument('--output_image_dir', type=str, default=OUTPUT_IMAGE_DIR_)
    parser.add_argument('--output_scene_dir', type=str, default=OUTPUT_SCENE_DIR_)
    parser.add_argument('--output_scene_file', type=str, default=OUTPUT_SCENE_FILE_)
    parser.add_argument('--output_question_file', type=str, default=OUTPUT_QUESTION_FILE_)
    parser.add_argument('--clear_all', type=str, default='True')

    args = parser.parse_args()
    image_seed_path = str(args.image_seed_path)
    random_agent_choice = eval(args.randac)
    clear_all = eval(args.clear_all)
    if clear_all:
        base = osp.dirname(osp.abspath(__file__)).replace('data_driven_limits', 'neural_render')
        images = base + '\images'
        scenes = base + '\scenes'
        shutil.rmtree(images)
        shutil.rmtree(scenes)
        os.mkdir(images)
        os.mkdir(scenes)
    if image_seed_path == 'None':
        image_seed_path = osp.dirname(osp.abspath(__file__)).replace('_driven_limits', '')
    output_image_dir = str(args.output_image_dir)
    output_scene_dir = str(args.output_scene_dir)
    output_scene_file = str(args.output_scene_file)
    output_question_file = str(args.output_question_file)
    initialize_paths(output_scene_dir=OUTPUT_SCENE_DIR_, output_image_dir=OUTPUT_IMAGE_DIR_, up_to_here=UP_TO_HERE_)
    filelocator = CacheFile(name=f"val_seed_{args.image_seed_id}", func=initialize_scene_seed,
                            func_kwargs={'scenes_path': image_seed_path, 'scene_id': args.image_seed_id})
    s = filelocator.get()
    generate_dataset(number_of_objects=args.noo, number_of_swaps=args.nos, image_seed=s,
                     agent_percentage_start=args.agentperstart, agent_percentage_end=args.agentperend,
                     random_agent_choice=random_agent_choice, split_name=args.split_name)
