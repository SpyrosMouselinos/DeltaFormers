import argparse
import math
import os
import sys

import tqdm
import itertools
import random
import time
import warnings
import os.path as osp
import numpy as np
import torch
import pickle
from neural_render.blender_render_utils.constants import SPLIT_, OUTPUT_IMAGE_DIR_, OUTPUT_SCENE_DIR_, \
    OUTPUT_SCENE_FILE_, \
    OUTPUT_QUESTION_FILE_, UP_TO_HERE_
from neural_render.blender_render_utils.helpers import render_image, initialize_paths
from utils.train_utils import MixCLEVR_HDF5, StateCLEVR

warnings.filterwarnings("ignore", category=UserWarning)
random.seed(666)
np.random.seed(666)

scenes_path = 'C:\\Users\\Guldan\\Desktop\\DeltaFormers\\data'
split ='val'


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
        actions_x = torch.LongTensor([f % self.n_dof  for f in actions])
        actions_y = torch.LongTensor([f // self.n_dof for f in actions])
        return actions_x, actions_y

    def register(self, example_id, action_id):
        self.registered_actions.append((example_id, action_id))
        return



def state2img(state, bypass=False, custom_index=0, delete_every=True, retry=False, perturbations_x=None, perturbations_y=None):
    wr = []
    images_to_be_rendered = n_possible_images = 1
    n_objects_per_image = state['types'][:10].sum().item()
    assert len(perturbations_x) == len(perturbations_y)
    assert len(perturbations_x) == n_objects_per_image

    key_light_jitter = fill_light_jitter = back_light_jitter = [0.5] * n_possible_images
    if retry:
        choices = [1, 0.5, 0.2, -0.2, -0.5, 0, -1]
    else:
        choices = [5]
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
            for object_idx in range(n_objects_per_image):
                tmp_x.append((state['object_positions'][object_idx].numpy()[0] * 3 + perturbations_x[object_idx]).clip(-2.98,2.98).item())
                tmp_y.append((state['object_positions'][object_idx].numpy()[1] * 3 + perturbations_y[object_idx]).clip(-2.98,2.98).item())
                tmp_z.append(state['object_positions'][object_idx].numpy()[2] * 360)
                tmp_colors.append(state['object_colors'][object_idx].item() - 1)
                tmp_shapes.append(state['object_shapes'][object_idx].item() - 1)
                tmp_materials.append(state['object_materials'][object_idx].item() - 1)
                tmp_sizes.append(state['object_sizes'][object_idx].item() - 1)

            if render_check(tmp_x, tmp_y, tmp_sizes, tmp_shapes):
                xs.append(tmp_x)
                ys.append(tmp_y)
                zs.append(tmp_z)
                colors.append(tmp_colors)
                shapes.append(tmp_shapes)
                materials.append(tmp_materials)
                sizes.append(tmp_sizes)
                questions.append(state['question'][image_idx])
                wr.append(image_idx)


                assembled_images = render_image(key_light_jitter=key_light_jitter, fill_light_jitter=fill_light_jitter,
                                                back_light_jitter=back_light_jitter, camera_jitter=camera_jitter,
                                                per_image_x=xs, per_image_y=ys, per_image_theta=zs, per_image_shapes=shapes,
                                                per_image_colors=colors, per_image_sizes=sizes,
                                                per_image_materials=materials,
                                                num_images=images_to_be_rendered, split='Defense2', start_idx=custom_index,
                                                workers=1)
                sys.exit(1)
                return assembled_images[0][1]
            else:
                return 0

if osp.exists(f'{scenes_path}/{split}_dataset.pt'):
    with open(f'{scenes_path}/{split}_dataset.pt', 'rb') as fin:
        info = pickle.load(fin)
        x = info['x'][830]
        y = info['y'][830]

# to_render = [5180, 5360, 5410, 5720]
# for i in to_render:
#     state2img(x[i], custom_index=i // 10)

# Mini Defense CLEVR #
# Image Seed Validation 000006  000083#
actionsx = torch.FloatTensor([0,0,0,0])
actionsy = torch.FloatTensor([0,0,0,0])
result = state2img(state=x, custom_index=0, delete_every=False, retry=False, perturbations_x=actionsx, perturbations_y=actionsy)
wizard = Wizard(2)
wizard.restart(2)
pbar = tqdm.tqdm(total=len(wizard.action_memory))
created = 0
for i in range(len(wizard.action_memory)):
    actionsx, actionsy = wizard.act(i)
    actionsx = actionsx - 3
    actionsy = actionsy - 3
    actionsx = actionsx / 3.0
    actionsy = actionsy / 3.0
    actionsx = torch.cat([torch.FloatTensor([0,0]), actionsx])
    actionsy = torch.cat([torch.FloatTensor([0,0]), actionsy])
    #print(f"Trying {actionsx}, {actionsy}")
    actionsx = torch.FloatTensor([0,0,0,0])
    actionsy = torch.FloatTensor([0,0,0,0])
    result = state2img(state=x, custom_index=i, delete_every=False, retry=False, perturbations_x=actionsx, perturbations_y=actionsy)
    if result == 1:
        wizard.register(i, [actionsx, actionsy])
        created += 1
    pbar.update(1)
    pbar.set_postfix({'Images Created so Far': created})
pbar.close()







def generate_images(max_images=2, batch_size=2, max_episodes=1, workers=1):
    global_generation_success_rate = 0

    ### MAIN LOOP ###
    episodes = 0
    start_time = time.time()
    while episodes < max_episodes:
        ### Move to Numpy Format
        key_light_jitter = [0.5, 0.5]
        fill_light_jitter = [0.5, 0.5]
        back_light_jitter = [0.5, 0.5]
        camera_jitter = [1, 1]
        per_image_shapes = [[0], [1]]
        per_image_colors = [[0], [1]]
        per_image_materials = [[0], [1]]
        per_image_sizes = [[0], [1]]

        per_image_x = [[0], [-0.3]]
        per_image_y = [[0], [-0.3]]
        per_image_theta = [[0], [0]]

        #### Render it ###
        attempted_images = render_image(key_light_jitter=key_light_jitter,
                                        fill_light_jitter=fill_light_jitter,
                                        back_light_jitter=back_light_jitter,
                                        camera_jitter=camera_jitter,
                                        per_image_shapes=per_image_shapes,
                                        per_image_colors=per_image_colors,
                                        per_image_materials=per_image_materials,
                                        per_image_sizes=per_image_sizes,
                                        per_image_x=per_image_x,
                                        per_image_y=per_image_y,
                                        per_image_theta=per_image_theta,
                                        num_images=batch_size,
                                        workers=workers,
                                        split=SPLIT_,
                                        assemble_after=False,
                                        start_idx=episodes * batch_size,
                                        )
        correct_images = [f[0] for f in attempted_images if f[1] == 1]
        global_generation_success_rate += len(correct_images)
        if global_generation_success_rate >= max_images:
            break
        episodes += 1
    end_time = time.time()
    ips = global_generation_success_rate / round(end_time - start_time, 2)
    duration = round(end_time - start_time, 2)
    print(f"Took {end_time - start_time} time for {global_generation_success_rate} images")
    print(f"Images per second {ips}")
    print(f"Generator Success Rate: {round(global_generation_success_rate / (max_episodes * batch_size), 2)}")
    return duration


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--ngpus', type=str, default=1)
#     parser.add_argument('--output_image_dir', type=str, default=OUTPUT_IMAGE_DIR_)
#     parser.add_argument('--output_scene_dir', type=str, default=OUTPUT_SCENE_DIR_)
#     parser.add_argument('--output_scene_file', type=str, default=OUTPUT_SCENE_FILE_)
#     parser.add_argument('--output_question_file', type=str, default=OUTPUT_QUESTION_FILE_)
#
#     args = parser.parse_args()
#     ngpus = str(args.ngpus)
#     output_image_dir = str(args.output_image_dir)
#     output_scene_dir = str(args.output_scene_dir)
#     output_scene_file = str(args.output_scene_file)
#     output_question_file = str(args.output_question_file)
#
#     initialize_paths(output_scene_dir=OUTPUT_SCENE_DIR_, output_image_dir=OUTPUT_IMAGE_DIR_, up_to_here=UP_TO_HERE_)
#     generate_images()
