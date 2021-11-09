import json
import os
import os.path as osp
import pickle
import random
import sys

import h5py
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import trange

sys.path.insert(0, osp.abspath('..'))
import yaml
from torch.utils.data import Dataset
from natsort import natsorted
import torch
from torch.optim.optimizer import Optimizer


def _print(something):
    print(something, flush=True)
    return


class BatchSizeScheduler:
    def __init__(self, train_ds, initial_bs, step_size, gamma, max_bs):
        self.train_ds = train_ds
        self.current_bs = initial_bs
        self.max_bs = max_bs
        self.step_size = step_size
        self.gamma = gamma
        self._current_steps = 0

    def reset(self):
        self._current_steps = 0
        return

    def step(self):
        if self.step_size != -1:
            self._current_steps += 1
            if self._current_steps % self.step_size == 0 and self._current_steps > 0:
                self.current_bs = min(self.current_bs * self.gamma, self.max_bs)
            return torch.utils.data.DataLoader(self.train_ds, batch_size=self.current_bs, shuffle=True)
        else:
            return torch.utils.data.DataLoader(self.train_ds, batch_size=self.current_bs, shuffle=True)

    def state_dict(self):
        info = {
            'current_bs': self.current_bs,
            'max_bs': self.max_bs,
            'step_size': self.step_size,
            'gamma': self.gamma,
            'current_steps': self._current_steps
        }
        return info

    def load_state_dict(self, state_dict):
        self.current_bs = state_dict['current_bs']
        self.max_bs = state_dict['max_bs']
        self.step_size = state_dict['step_size']
        self.gamma = state_dict['gamma']
        self._current_steps = state_dict['current_steps']


def image_finder(clvr_path='data/', mode='val'):
    ### Discover and return all available images ###
    good_images = []
    available = os.listdir(clvr_path + f'/images/{mode}')
    for candidate in available:
        if mode in candidate and candidate.endswith('.png'):
            good_images.append(candidate)
    return natsorted(good_images)


def scene_parser(scenes_path='data/', mode='val'):
    with open(scenes_path + f'/CLEVR_{mode}_scenes.json', 'r') as fin:
        parsed_json = json.load(fin)
        scenes = parsed_json['scenes']
    return scenes


def question_parser(questions_path='data/', mode='val'):
    with open(questions_path + f'/CLEVR_{mode}_questions.json', 'r') as fin:
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
    p = question['program']
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
                return image_index, None, None, None, None
            elif '?' in word or ';' in word:
                tokenized_q.append(q2index[word[:-1]])
                tokenized_q.append(q2index[';'])
            else:
                try:
                    tokenized_q.append(q2index[word])
                except KeyError:
                    tokenized_q.append(3)
        q = torch.LongTensor(tokenized_q + [0] * (50 - len(tokenized_q))).view(50)
    if a2index is None:
        pass
    else:
        a = torch.LongTensor([a2index[a] - 4])

    return image_index, len(tokenized_q), q, a, p


def scene_image_matcher(split, translation, q2index, a2index, scenes_path='data/', questions_path='data/'):
    ### All scenes ###
    scenes = scene_parser(scenes_path, split)

    ### All questions ###
    questions = question_parser(questions_path, split)

    x_samples = []
    y_samples = []
    p_samples = []
    question_counter = 0
    for scene_counter in trange(len(scenes)):
        image_index_scene, n_objects, object_positions, object_colors, object_shapes, object_materials, object_sizes = \
            single_scene_translator(scene=scenes[scene_counter], translation=translation)
        while question_counter < len(questions):
            image_index_question, n_tokens, q, a, p = single_question_parser(questions[question_counter],
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
                p_samples.append(p)
                # Increment and Loop #
                question_counter += 1
            else:
                # Question is for the next image #
                break
    return x_samples, y_samples, p_samples


def visual_image_matcher(split, q2index, a2index, clvr_path='data/', questions_path='data/'):
    ### All images ###
    images = image_finder(clvr_path, split)

    ### All questions ###
    questions = question_parser(questions_path, split)

    x_samples = []
    y_samples = []
    p_samples = []
    question_counter = 0
    for scene_counter in trange(len(images)):
        image_index_scene = int(images[scene_counter].split('.png')[0].split(f'{split}_')[-1])
        while question_counter < len(questions):
            image_index_question, n_tokens, q, a, p = single_question_parser(questions[question_counter],
                                                                             word_replace_dict={'True': 'yes',
                                                                                                'False': 'no'},
                                                                             q2index=q2index,
                                                                             a2index=a2index)
            # Bad question Move on #
            if q is None and a is None:
                question_counter += 1
                continue

            if image_index_scene == image_index_question:
                x_samples.append({'image_filename': images[scene_counter],
                                  'question': q,
                                  })
                y_samples.append(a)
                p_samples.append(p)
                # Increment and Loop #
                question_counter += 1
            elif image_index_scene < image_index_question:
                # Question is for the next image #
                break
            elif image_index_scene > image_index_question:
                # Question is for a previous image #
                question_counter += 1
    return x_samples, y_samples, p_samples


class StateCLEVR(Dataset):
    """CLEVR dataset made from Scene States."""

    def __init__(self, config=None, split='val', scenes_path='data/', questions_path='data/', clvr_path=None,
                 use_cache=False, return_program=False, effective_range=None, randomize_range=False,
                 effective_range_offset=0):
        if randomize_range:
            if effective_range is not None:
                effective_range_offset = random.randint(0, 140_000 - effective_range)
            else:
                effective_range_offset = 0
        else:
            effective_range_offset = effective_range_offset
        print(f"Effective Range Offset: {effective_range_offset}", flush=True)
        self.return_program = return_program
        if osp.exists(f'{scenes_path}/{split}_dataset.pt'):
            with open(f'{scenes_path}/{split}_dataset.pt', 'rb') as fin:
                info = pickle.load(fin)
            self.split = info['split']
            self.translation = info['translation']
            self.q2index = info['q2index']
            self.a2index = info['a2index']
            if effective_range is None:
                self.x = info['x']
                self.y = info['y']
            else:
                self.x = info['x'][int(effective_range_offset):int(effective_range) + int(effective_range_offset)]
                self.y = info['y'][int(effective_range_offset):int(effective_range) + int(effective_range_offset)]

            if self.return_program:
                try:
                    if effective_range is None:
                        self.p = info['p']
                    else:
                        self.p = info['p'][
                                 int(effective_range_offset):int(effective_range) + int(effective_range_offset)]
                except KeyError:
                    print("Dataset loaded without program!\n")
                    self.return_program = False
            print("Dataset loaded succesfully!\n")
        else:
            with open(osp.dirname(osp.dirname(__file__)) + '/translation_tables.yaml', 'r') as fin:
                translation = yaml.load(fin, Loader=yaml.FullLoader)['translation']
            with open(f'{questions_path}/vocab.json', 'r') as fin:
                parsed_json = json.load(fin)
                q2index = parsed_json['question_token_to_idx']
                a2index = parsed_json['answer_token_to_idx']

            self.split = split
            # self.config = config
            self.translation = translation
            self.q2index = q2index
            self.a2index = a2index
            x, y, p = scene_image_matcher(self.split, self.translation, self.q2index, self.a2index, scenes_path,
                                          questions_path)
            self.x = x
            self.y = y
            self.p = p
            print("Dataset loaded succesfully!...Saving\n")
            info = {
                'split': self.split,
                'translation': self.translation,
                'q2index': self.q2index,
                'a2index': self.a2index,
                'x': self.x,
                'y': self.y,
                'p': self.p
            }
            with open(f'{scenes_path}/{self.split}_dataset.pt', 'wb') as fout:
                pickle.dump(info, fout)

        ### Rectify Programs ###
        if self.return_program:
            new_p = []
            padding = 'P'
            for entry in self.p:
                entry = str(entry)
                new_entry = padding * (2500 - len(entry)) + entry
                new_p.append(new_entry)
            self.p = new_p

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.return_program:
            return self.x[idx], self.y[idx], self.p[idx]
        else:
            return self.x[idx], self.y[idx]


class ImageCLEVR_HDF5(Dataset):
    """CLEVR dataset made from Images in HDF5 format."""

    def __init__(self, config=None, split='val', clvr_path='data/', questions_path='data/',
                 scenes_path=None, use_cache=False, return_program=False, effective_range=None, output_shape=None,
                 randomize_range=False,
                 effective_range_offset=0, prior_shuffle=False, indicies=None):
        self.sb = indicies
        if randomize_range:
            if effective_range is not None:
                print(effective_range)
                effective_range_offset = random.randint(0, max(0, 15560 - int(effective_range) - 1))
            else:
                effective_range_offset = 0
        else:
            effective_range_offset = effective_range_offset
        if output_shape is None:
            print("Assuming Image outputs of size 128x128")
            self.shape = 128
        else:
            print(f"Assuming Image outputs of size {output_shape}x{output_shape}")
            self.shape = output_shape
        self.return_program = return_program
        self.clvr_path = clvr_path
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        if split == 'train':
            self.transform = transforms.Compose([transforms.Pad(8),
                                                 transforms.RandomCrop((self.shape, self.shape)),
                                                 transforms.RandomRotation(2.8),  # .05 rad
                                                 transforms.ToTensor(),
                                                 normalize])
        else:
            self.transform = transforms.Compose(
                [transforms.Resize((self.shape, self.shape)), transforms.ToTensor(), normalize])
        if osp.exists(f'{clvr_path}/{split}_image_dataset_{self.shape}.pt'):
            with open(f'{clvr_path}/{split}_image_dataset_{self.shape}.pt', 'rb') as fin:
                info = pickle.load(fin)
            self.split = info['split']
            self.q2index = info['q2index']
            self.a2index = info['a2index']
            # if effective_range is None:
            #     if prior_shuffle:
            #         if self.indices is None:
            #             all_indexes = list(enumerate(list(range(len(info['x'])))))
            #             random.shuffle(all_indexes)
            #             self.indices, _ = zip(*all_indexes)
            #             self.indices = list(self.indices)
            #         self.x = [info['x'][j] for j in self.indices][int(effective_range_offset):]
            #         self.y = [info['y'][j] for j in self.indices][int(effective_range_offset):]
            #     else:
            #         if self.indices is None:
            #             all_indexes = list(enumerate(list(range(len(info['x'])))))
            #             self.indices, _ = zip(*all_indexes)
            #             self.indices = list(self.indices)
            #         self.x = [info['x'][j] for j in self.indices]
            #         self.y = [info['y'][j] for j in self.indices]
            # else:
            #     if prior_shuffle:
            #         if self.indices is None:
            #             all_indexes = list(enumerate(list(range(len(info['x'])))))
            #             random.shuffle(all_indexes)
            #             self.indices, _ = zip(*all_indexes)
            #             self.indices = list(self.indices)
            #         self.x = [info['x'][j] for j in self.indices]
            #         self.y = [info['y'][j] for j in self.indices]
            if effective_range is None and effective_range_offset == 0:
                x_ = info['x']
                y_ = info['y']
            else:
                # x_ = info['x']
                # y_ = info['y']
                # x_ = self.interleave_list(info['x'], skip_limit=len(info['x']) // 6, number_of_limits=6)
                # y_ = self.interleave_list(info['y'], skip_limit=len(info['y']) // 6, number_of_limits=6)
                x_, sb = self.shuffle(info['x'], by=self.sb)
                y_, _ = self.shuffle(info['y'], by=sb)
                if self.sb is None:
                   self.sb = sb
            if effective_range is None:
                effective_range = len(x_)
            else:
                effective_range = effective_range * len(x_)

            if effective_range_offset is None:
                effective_range_offset = 0
            else:
                effective_range_offset = effective_range_offset * len(x_)

            self.x = x_[int(effective_range_offset):int(effective_range) + int(effective_range_offset)]
            self.y = y_[int(effective_range_offset):int(effective_range) + int(effective_range_offset)]
            if self.return_program:
                try:
                    if effective_range is None:
                        self.p = info['p']
                    else:
                        self.p = info['p'][
                                 int(effective_range_offset):int(effective_range) + int(effective_range_offset)]
                except KeyError:
                    _print("Dataset loaded without program!\n")
                    self.return_program = False
            _print(f"Dataset {self.shape} loaded succesfully!\n")
        else:
            self.split = split
            with open(f'{questions_path}/vocab.json', 'r') as fin:
                parsed_json = json.load(fin)
                self.q2index = parsed_json['question_token_to_idx']
                self.a2index = parsed_json['answer_token_to_idx']
            x, y, p = visual_image_matcher(split, self.q2index, self.a2index, clvr_path, questions_path)
            self.x = x
            self.y = y
            self.p = p
            _print("Dataset matched succesfully!\n")
            info = {
                'split': self.split,
                'q2index': self.q2index,
                'a2index': self.a2index,
                'x': self.x,
                'y': self.y,
                'p': self.p
            }
            with open(f'{clvr_path}/{self.split}_image_dataset_{self.shape}.pt', 'wb') as fout:
                pickle.dump(info, fout)
        if osp.exists(f'{clvr_path}/{split}_images_{self.shape}.h5'):
            self.hdf5_file = np.array(h5py.File(f'{clvr_path}/{split}_images_{self.shape}.h5', 'r')['image']).astype(
                "uint8")
            self.n_images = self.hdf5_file.shape[0]
            _print("Image HDF5 loaded succesfully!\n")
        else:
            available_images = natsorted(os.listdir(self.clvr_path + f'/images/{self.split}/'))
            image_train_shape = (len(available_images), self.shape, self.shape, 3)

            f = h5py.File(f'{clvr_path}/{split}_images_{self.shape}.h5', mode='w')
            f.create_dataset("image", image_train_shape, h5py.h5t.STD_U8BE)

            for i, img_addr in enumerate(available_images):
                image = Image.open(self.clvr_path + f'/images/{split}/{img_addr}').convert('RGB').resize(
                    (self.shape, self.shape), 3)
                f["image"][i] = image
            f.close()
            _print("Image HDF5 written succesfully!\n")
            self.hdf5_file = np.array(h5py.File(f'{clvr_path}/{split}_images_{self.shape}.h5', 'r')['image']).astype(
                "uint8")
            self.n_images = self.hdf5_file.shape[0]
            _print("Image HDF5 loaded succesfully!\n")

        ### Rectify Programs ###
        if self.return_program:
            new_p = []
            padding = 'P'
            for entry in self.p:
                entry = str(entry)
                new_entry = padding * (2500 - len(entry)) + entry
                new_p.append(new_entry)
            self.p = new_p

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        question = self.x[idx]['question']
        current_image_fn = self.x[idx]['image_filename']
        current_image_index = int(current_image_fn.split('.png')[0].split(f'{self.split}_')[-1])
        image_data = Image.fromarray(self.hdf5_file[current_image_index], 'RGB')
        image = self.transform(image_data)
        answer = self.y[idx]
        # program = self.p[idx]

        return {'image': image, 'question': question}, answer

    def interleave_list(self, ql, skip_limit=1782, number_of_limits=8):
        new_l = []
        for i in range(0, skip_limit):
            for j in range(0, number_of_limits):
                new_l.append(ql[i + j * skip_limit])
        return new_l



    def shuffle(self, l, by=None):
        if by is None:
            x = list(enumerate(l))
            random.shuffle(x)
            indices, l = zip(*x)
            return l, indices
        else:
            x = [l[index] for index in by]
            return x, by


class MixCLEVR_HDF5(Dataset):
    def __init__(self, config=None, split='val', scenes_path='data/', clvr_path='data/', questions_path='data/',
                 use_cache=False, return_program=False, effective_range=None, output_shape=None, randomize_range=False,
                 effective_range_offset=0):
        if randomize_range:
            if effective_range is not None:
                effective_range_offset = random.randint(0, 140_000 - effective_range)
                effective_range_offset = (effective_range_offset // 10) * 10
            else:
                effective_range_offset = 0
        else:
            effective_range_offset = effective_range_offset
        print(f"Effective Range Offset: {effective_range_offset}", flush=True)
        self.effective_range_offset = effective_range_offset
        self.return_program = return_program
        self.split = split
        self.clvr_path = clvr_path
        self.state_ds = StateCLEVR(config=None, split=split, scenes_path=scenes_path, questions_path=questions_path,
                                   clvr_path=clvr_path, use_cache=use_cache, return_program=True,
                                   effective_range=effective_range, effective_range_offset=effective_range_offset)
        self.image_ds = ImageCLEVR_HDF5(config=None, split=split, clvr_path=clvr_path, questions_path=questions_path,
                                        scenes_path=scenes_path, use_cache=use_cache, return_program=return_program,
                                        effective_range=effective_range, output_shape=output_shape,
                                        effective_range_offset=effective_range_offset)

        if len(self.state_ds) != len(self.image_ds):
            print("Oops")
        self.max_data_length = min(len(self.state_ds), len(self.image_ds))

        # assert len(self.state_ds) == len(self.image_ds)

    def __len__(self):
        return len(self.image_ds.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if idx > self.max_data_length:
            raise StopIteration
        full_state = self.state_ds.x[idx]
        assert (full_state['question'] == self.image_ds.x[idx]['question']).all()
        current_image_fn = self.image_ds.x[idx]['image_filename']
        current_image_index = int(current_image_fn.split('.png')[0].split(f'{self.split}_')[-1])
        image_data = Image.fromarray(self.image_ds.hdf5_file[current_image_index], 'RGB')
        image = self.image_ds.transform(image_data)
        answer = self.state_ds.y[idx]
        assert answer == self.image_ds.y[idx]
        program = self.state_ds.p[idx]

        return (full_state, {'image': image, 'question': full_state['question']}), answer, program


class ImageCLEVR(Dataset):
    """CLEVR dataset made from Images."""

    def __init__(self, config=None, split='val', use_cache=False, clvr_path='data/', questions_path='data/',
                 scenes_path=None):
        self.use_cache = use_cache
        self.clvr_path = clvr_path
        if split == 'train':
            self.transform = transforms.Compose([transforms.Resize((128, 128)),
                                                 transforms.Pad(8),
                                                 transforms.RandomCrop((128, 128)),
                                                 transforms.RandomRotation(2.8),  # .05 rad
                                                 transforms.ToTensor()])
        else:
            self.transform = transforms.Compose([transforms.Resize((128, 128)),
                                                 transforms.ToTensor()])
        if osp.exists(f'{clvr_path}/{split}_image_dataset.pt'):
            with open(f'{clvr_path}/{split}_image_dataset.pt', 'rb') as fin:
                info = pickle.load(fin)
            self.split = info['split']
            self.q2index = info['q2index']
            self.a2index = info['a2index']
            self.x = info['x']
            self.y = info['y']
            _print("Dataset loaded succesfully!\n")
        else:
            self.split = split
            with open(f'{questions_path}/vocab.json', 'r') as fin:
                parsed_json = json.load(fin)
                self.q2index = parsed_json['question_token_to_idx']
                self.a2index = parsed_json['answer_token_to_idx']
            x, y = visual_image_matcher(split, self.q2index, self.a2index, clvr_path, questions_path)
            self.x = x
            self.y = y
            _print("Dataset loaded succesfully!...Saving\n")
            info = {
                'split': self.split,
                'q2index': self.q2index,
                'a2index': self.a2index,
                'x': self.x,
                'y': self.y
            }
            with open(f'{clvr_path}/{self.split}_image_dataset.pt', 'wb') as fout:
                pickle.dump(info, fout)

        self.cached_images = {}

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        current_image_fn = self.x[idx]['image_filename']
        question = self.x[idx]['question']
        if self.use_cache:
            if current_image_fn not in self.cached_images:
                image = Image.open(self.clvr_path + f'/images/{self.split}/{current_image_fn}').convert('RGB')
                image = self.transform(image)
                self.cached_images.update({current_image_fn: image})
            else:
                image = self.cached_images[current_image_fn]
        else:
            image = Image.open(self.clvr_path + f'/images/{self.split}/{current_image_fn}').convert('RGB')
            image = self.transform(image)

        answer = self.y[idx]

        return {'image': image, 'question': question}, answer


class SGD_GC(Optimizer):

    def __init__(self, params, lr, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD_GC, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD_GC, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)

                # GC operation for Conv layers and FC layers
                if len(list(d_p.size())) > 1:
                    d_p.add_(-d_p.mean(dim=tuple(range(1, len(list(d_p.size())))), keepdim=True))

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-group['lr'], d_p)

        return loss
