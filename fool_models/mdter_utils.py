import json
import os
import torch
import torch.nn as nn
import yaml
from fool_models.mdetr import MDetrWrapper

from neural_render.blender_render_utils.constants import find_platform_slash
from utils.train_utils import ImageCLEVR_HDF5

PLATFORM_SLASH = find_platform_slash()
UP_TO_HERE_ = PLATFORM_SLASH.join(os.path.abspath(__file__).split(PLATFORM_SLASH)[:-2]).replace(PLATFORM_SLASH, '/')


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
        # (_, iq), answers, _ = batch
        iq, answers = batch
        image = iq['image']
        questions_var = iq['question']
        preds = model(image, questions_var)

        for item in preds:
            # For line 684 in reinforce_train.py
            final_preds.append(item)

        num_correct += (torch.LongTensor(preds) == (answers.squeeze() - 4)).sum()
        num_samples += preds.size(0)
        if num_samples % 1000 == 0:
            print(f'Ran {num_samples} samples at {float(num_correct) / num_samples} accuracy')

    acc = float(num_correct) / num_samples
    print('[MDetr] Got %d / %d = %.2f correct' % (num_correct, num_samples, 100 * acc))
    return final_preds


loader = load_loader()
model = load_mdetr()
boo = inference_with_mdetr(loader, model, None)
