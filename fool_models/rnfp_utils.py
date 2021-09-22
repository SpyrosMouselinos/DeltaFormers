import json
import os

import torch
import torch.nn as nn
import yaml

from modules.embedder import DeltaRNFP
from neural_render.blender_render_utils.constants import find_platform_slash
from utils.train_utils import ImageCLEVR_HDF5

PLATFORM_SLASH = find_platform_slash()
UP_TO_HERE_ = PLATFORM_SLASH.join(os.path.abspath(__file__).split(PLATFORM_SLASH)[:-2]).replace(PLATFORM_SLASH, '/')


def invert_dict(d):
    return {value: key for (key, value) in d.items()}


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
        print(f"Your model achieves {round(checkpoint['val_loss'], 4)} validation loss\n")
    except:
        print("No Loss registered")
    return model


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

def kwarg_dict_to_device(data_obj, device):
    if device == 'cpu':
        return data_obj
    cpy = {}
    for key, _ in data_obj.items():
        cpy[key] = data_obj[key].to(device)
    return cpy

def load_rnfp(model_path=None):
    config = f'{UP_TO_HERE_}/results/experiment_fp/config.yaml'
    with open(config, 'r') as fin:
        config = yaml.load(fin, Loader=yaml.FullLoader)
    model = DeltaRNFP(config)
    if model_path is None:
        model = load(path=f'{UP_TO_HERE_}/results/experiment_fp/mos_epoch_219.pt', model=model)
    else:
        model = load(path=model_path, model=model)
    model.to('cuda')
    model.eval()
    return model


def load_loader():
    val_set = ImageCLEVR_HDF5(config=None, split='val',
                              clvr_path=f'{UP_TO_HERE_}/data',
                              questions_path=f'{UP_TO_HERE_}/data',
                              scenes_path=f'{UP_TO_HERE_}/data',
                              use_cache=False,
                              return_program=False,
                              effective_range=None, output_shape=128)

    val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=10,
                                                 num_workers=0, shuffle=False, drop_last=False)
    return val_dataloader


def load_resnet_backbone():
    raise ValueError("Not required in RNFP!\n")


def inference_with_rnfp(loader=None, model=None, resnet_extractor=None, evaluator=False):
    model.eval()
    num_correct, num_samples = 0, 0
    final_preds = []
    print(f"Testing for {len(loader)} samples")
    print()
    for batch in loader:
        if evaluator:
            iq, answer = batch
        else:
            (_, iq), answer, _ = batch

        iq = kwarg_dict_to_device(iq, 'cuda')
        scores, _ , _ = model(**iq)

        _, preds = scores.data.cpu().max(1)
        for item in preds.detach().cpu().numpy():
            final_preds.append(item)
        num_correct += (preds == (answer.squeeze())).sum()
        num_samples += preds.size(0)
        if num_samples % 1000 == 0:
            print(f'Ran {num_samples} samples at {float(num_correct) / num_samples} accuracy')

    acc = float(num_correct) / num_samples
    print('Got %d / %d = %.2f correct' % (num_correct, num_samples, 100 * acc))
    return final_preds


#model = load_rnfp()
#loader = load_loader()
#inference_with_rnfp(loader=None, model=model, resnet_extractor=None)
