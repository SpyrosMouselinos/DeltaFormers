import math
import os.path as osp
import pickle
import itertools
import os.path

import torch
from neural_render.blender_render_utils.helpers import render_image

filepos = osp.dirname(osp.abspath(__file__))


class CacheFile:
    def __init__(self, name, func, func_kwargs):
        self.name = name
        if os.path.exists(f"{filepos}\\.cache"):
            pass
        else:
            os.mkdir(f"{filepos}\\.cache")

        if os.path.exists(f"{filepos}\\.cache\\{name}.cpt"):
            with open(f"{filepos}\\.cache\\{name}.cpt", 'rb') as handle:
                self.result = pickle.load(handle)
        else:
            data = func(**func_kwargs)
            self.update(data)
            self.result = data

    def update(self, data):
        with open(f"{filepos}\\.cache\\{self.name}.cpt", 'wb') as handle:
            pickle.dump(data, handle)
            print(f"Storing {self.name} into local cache...\n")

    def get(self):
        return self.result


def rotate(l, n):
    return l[n:] + l[:n]


def swap(n, tmp_x, tmp_y, tmp_z, tmp_sizes, tmp_shapes, tmp_colors, tmp_materials, pad=False):
    # Mini Swap #
    if n in [0, 1, 2]:
        if pad:
            tmp_x = [tmp_x[0], tmp_x[1]]
            tmp_y = [tmp_y[0], tmp_y[1]]
            tmp_z = [tmp_z[0], tmp_z[1]]
        else:
            tmp_x = [tmp_x[0], tmp_x[1], tmp_x[2]]
            tmp_y = [tmp_y[0], tmp_y[1], tmp_y[2]]
            tmp_z = [tmp_z[0], tmp_z[1], tmp_z[2]]
    else:
        tmp_x = [tmp_x[0], tmp_x[2], tmp_x[1]]
        tmp_y = [tmp_y[0], tmp_y[2], tmp_y[1]]
        tmp_z = [tmp_z[0], tmp_z[2], tmp_z[1]]
        n = n - 3
    tmp_x_1 = rotate(tmp_x, n)
    tmp_y_1 = rotate(tmp_y, n)
    tmp_z_1 = rotate(tmp_z, n)
    if pad:
        tmp_sizes_1 = rotate(tmp_sizes[0:2], 0)
        tmp_shapes_1 = rotate(tmp_shapes[0:2], 0)
        tmp_colors_1 = rotate(tmp_colors[0:2], 0)
        tmp_materials_1 = rotate(tmp_materials[0:2], 0)
    else:
        tmp_sizes_1 = rotate(tmp_sizes, 0)
        tmp_shapes_1 = rotate(tmp_shapes, 0)
        tmp_colors_1 = rotate(tmp_colors, 0)
        tmp_materials_1 = rotate(tmp_materials, 0)
    return tmp_x_1, tmp_y_1, tmp_z_1, tmp_sizes_1, tmp_shapes_1, tmp_colors_1, tmp_materials_1


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
        actions_x = torch.LongTensor([f % self.n_dof for f in actions])
        actions_y = torch.LongTensor([f // self.n_dof for f in actions])
        return actions_x, actions_y

    def register(self, example_id, action_id):
        self.registered_actions.append((example_id, action_id))
        return


def state2img(state,
              bypass=False,
              custom_index=0,
              retry=False,
              perturbations_x=None,
              perturbations_y=None, split_name='Limits', swaps=1, pad=True):
    if pad:
        experiment_name = '2it'
    else:
        experiment_name = '3it'
    split_name = split_name + f'_{experiment_name}'
    wr = []
    images_to_be_rendered = n_possible_images = 1
    n_objects_per_image = state['types'][:10].sum().item()

    key_light_jitter = fill_light_jitter = back_light_jitter = [0.5] * n_possible_images
    if retry:
        choices = [1, 0.5, 0.2, -0.2, -0.5, 0, -1]
    else:
        choices = [0]
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
                tmp_x.append(
                    (state['object_positions'][object_idx].numpy()[0] * 3 + perturbations_x[object_idx]).clip(-2.98,
                                                                                                              2.98).item())
                tmp_y.append(
                    (state['object_positions'][object_idx].numpy()[1] * 3 + perturbations_y[object_idx]).clip(-2.98,
                                                                                                              2.98).item())
                tmp_z.append(state['object_positions'][object_idx].numpy()[2] * 360)
                tmp_colors.append(state['object_colors'][object_idx].item() - 1)
                tmp_shapes.append(state['object_shapes'][object_idx].item() - 1)
                tmp_materials.append(state['object_materials'][object_idx].item() - 1)
                tmp_sizes.append(state['object_sizes'][object_idx].item() - 1)

            assembled_images = 0
            if render_check(tmp_x, tmp_y, tmp_sizes, tmp_shapes) or bypass:
                for i in range(0, swaps+1):
                    tmp_x_, tmp_y_, tmp_z_, tmp_sizes_, tmp_shapes_, tmp_colors_, tmp_materials_ = swap(i, tmp_x, tmp_y,
                                                                                                        tmp_z,
                                                                                                        tmp_sizes,
                                                                                                        tmp_shapes,
                                                                                                        tmp_colors,
                                                                                                        tmp_materials, pad=pad)
                    xs.append(tmp_x_)
                    ys.append(tmp_y_)
                    zs.append(tmp_z_)
                    colors.append(tmp_colors_)
                    shapes.append(tmp_shapes_)
                    materials.append(tmp_materials_)
                    sizes.append(tmp_sizes_)
                    wr.append(image_idx)

                    assembled_images_ = render_image(key_light_jitter=key_light_jitter,
                                                     fill_light_jitter=fill_light_jitter,
                                                     back_light_jitter=back_light_jitter, camera_jitter=camera_jitter,
                                                     per_image_x=xs, per_image_y=ys, per_image_theta=zs,
                                                     per_image_shapes=shapes,
                                                     per_image_colors=colors, per_image_sizes=sizes,
                                                     per_image_materials=materials,
                                                     num_images=images_to_be_rendered, split=f'{split_name}_{i}',
                                                     start_idx=custom_index,
                                                     workers=1)
                    xs = []
                    ys = []
                    zs = []
                    colors = []
                    shapes = []
                    materials = []
                    sizes = []
                    wr = []
                    assembled_images += assembled_images_[0][1]

                return assembled_images
            else:
                return 0


__all__ = ['rotate', 'swap', 'render_check', 'Wizard', 'state2img', 'CacheFile']
