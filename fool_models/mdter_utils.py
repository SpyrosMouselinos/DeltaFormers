import json
import os
import torch
import torch.nn as nn
import yaml
from fool_models.mdetr import MDetrWrapper
from skimage.color import rgba2rgb
from skimage.io import imread
import numpy as np
from skimage.transform import resize as imresize

from neural_render.blender_render_utils.constants import find_platform_slash
from utils.train_utils import ImageCLEVR_HDF5

PLATFORM_SLASH = find_platform_slash()
UP_TO_HERE_ = PLATFORM_SLASH.join(os.path.abspath(__file__).split(PLATFORM_SLASH)[:-2]).replace(PLATFORM_SLASH, '/')
def invert_dict(d):
    return {value: key for (key, value) in d.items()}


SPECIAL_TOKENS = {
    '<NULL>': 0,
    '<START>': 1,
    '<END>': 2,
    '<UNK>': 3,
}

with open(f'{UP_TO_HERE_}/fool_models/resources/vocab.json', 'r') as fin:
    data = json.loads(fin.read())

question_token_to_idx = data['question_token_to_idx']
program_token_to_idx = data['program_token_to_idx']
answer_token_to_idx = data['answer_token_to_idx']

idx_to_question_token = invert_dict(question_token_to_idx)
idx_to_program_token = invert_dict(program_token_to_idx)
idx_to_answer_token = invert_dict(answer_token_to_idx)

data['question_idx_to_token'] = idx_to_question_token
data['program_idx_to_token'] = idx_to_program_token
data['answer_idx_to_token'] = idx_to_answer_token

def load_loader():
    val_set = ImageCLEVR_HDF5(config=None, split='val',
                              clvr_path=f'{UP_TO_HERE_}/data',
                              questions_path=f'{UP_TO_HERE_}/data',
                              scenes_path=f'{UP_TO_HERE_}/data',
                              use_cache=False,
                              return_program=False,
                              effective_range=10, output_shape=224)

    val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=3,
                                                 num_workers=0, shuffle=False, drop_last=False)
    return val_dataloader


def load_resnet_backbone():
    raise ValueError("MDetr does not need ResNet...\n")


def load_mdetr():
    model = MDetrWrapper()
    model.load_mdetr()
    return model


def inference_with_mdetr(loader=None, model=None, resnet_extractor=None):
    final_preds = []
    num_correct, num_samples = 0, 0
    print(f"Testing for {len(loader)} samples")
    print()
    for batch in loader:
        (_, iq), answers, _ = batch
        #iq, answers = batch
        image = iq['image']
        questions_var = iq['question']
        preds, _, _ = model(image, questions_var)

        for item in preds:
            # For line 684 in reinforce_train.py
            final_preds.append(item - 4)

        num_correct += (torch.LongTensor(preds) == (answers.squeeze() + 4)).sum()
        num_samples += preds.size(0)
        if num_samples % 1000 == 0:
            print(f'Ran {num_samples} samples at {float(num_correct) / num_samples} accuracy')

    acc = float(num_correct) / num_samples
    print('[MDetr] Got %d / %d = %.2f correct' % (num_correct, num_samples, 100 * acc))
    return final_preds


def sinference_with_mdetr(model=None, loader=None, resnet_extractor=None):
    img = imread('C:\\Users\\Guldan\\Desktop\\DeltaFormers\\data\\images\\val\\CLEVR_val_000000.png')
    question = 'How many gray cubes are there?'
    #question_l = [1, 10, 85, 14, 25, 30, 64, 66, 84, 74, 75, 21, 84, 45, 86, 5, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    try:
        img = rgba2rgb(img)
    except:
        pass
    img = imresize(img, (224,224))
    img = img.astype('float32')
    img = img.transpose(2, 0, 1)[None]
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.224]).reshape(1, 3, 1, 1)
    img = (img - mean) / std
    img_var = torch.FloatTensor(img).to('cuda')
    questions_var = torch.LongTensor([question_token_to_idx[f] for f in question[:-1].split(' ')]).to('cuda')
    questions_var = questions_var.unsqueeze(0)
    preds, _, _ = model(img_var, questions_var, True)
    try:
        print(idx_to_answer_token[preds[0].cpu().item()])
    except:
        print(preds)
    return preds

#loader = load_loader()
model = load_mdetr()
boo = sinference_with_mdetr(model, None, None)
