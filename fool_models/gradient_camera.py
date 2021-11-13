import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import filters
from skimage.color import rgba2rgb
from skimage.io import imread
from skimage.transform import resize as imresize

from fool_models.film_utils import load_film
from fool_models.film_utils import load_resnet_backbone as load_film_resnet
from fool_models.iep_utils import load_iep
from fool_models.iep_utils import load_resnet_backbone as load_iep_resnet
from fool_models.stack_attention_utils import load_cnn_sa
from fool_models.stack_attention_utils import load_resnet_backbone as load_sa_resnet
from fool_models.stack_attention_utils import question_token_to_idx, answer_token_to_idx, idx_to_answer_token
from fool_models.rnfp_utils import load_rnfp


def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a])


def normalize(x):
    x = x - x.min()
    if x.max() > 0:
        x = x / x.max()
    return x


def getAttMap(img, attn_map, blur=True):
    if blur:
        attn_map = filters.gaussian_filter(attn_map, 0.02 * max(img.shape[:2]))
    attn_map = normalize(attn_map)
    cmap = plt.get_cmap('jet')
    attn_map_c = np.delete(cmap(attn_map), 3, 2)
    attn_map = 1 * (1 - attn_map ** 0.7).reshape(attn_map.shape + (1,)) * img + \
               (attn_map ** 0.7).reshape(attn_map.shape + (1,)) * attn_map_c
    return attn_map


def viz_attn(img, attn_map, blur=True):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img)
    axes[1].imshow(getAttMap(img, attn_map, blur))
    for ax in axes:
        ax.axis("off")
    plt.show()


class Hook:
    """Attaches to a module and records its activations and gradients."""

    def __init__(self, module: nn.Module):
        self.data = None
        self.hook = module.register_forward_hook(self.save_grad)

    def save_grad(self, module, input, output):
        self.data = output
        output.requires_grad_(True)
        output.retain_grad()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.hook.remove()

    @property
    def activation(self) -> torch.Tensor:
        return self.data

    @property
    def gradient(self) -> torch.Tensor:
        return self.data.grad


def gradCAM(
        model,
        inputs,
        target,
        layer=None):
    def pad_with(vector, pad_width, iaxis, kwargs):
        pad_value = kwargs.get('padder', 10)
        vector[:pad_width[0]] = pad_value
        vector[-pad_width[1]:] = pad_value

    if inputs['questions'].grad is not None:
        inputs['questions'].grad.data.zero_()

    if inputs['feats'].grad is not None:
        inputs['feats'].grad.data.zero_()

    # Disable gradient settings.
    requires_grad = {}
    if isinstance(model, tuple):
        for name, param in model[0].named_parameters():
            requires_grad[name] = param.requires_grad
            param.requires_grad_(False)
        for name, param in model[1].named_parameters():
            requires_grad[name] = param.requires_grad
            param.requires_grad_(False)
    else:
        for name, param in model.named_parameters():
            requires_grad[name] = param.requires_grad
            param.requires_grad_(False)

    # Attach a hook to the model at the desired layer.
    assert isinstance(layer, nn.Module)
    with Hook(layer) as hook:
        # Do a forward and backward pass.
        if isinstance(model, tuple):
            try:
                programs_pred = model[0].reinforce_sample(inputs['questions'],temperature=1,argmax=True)
            except:
                programs_pred = model[0](inputs['questions'])
            output = model[1](inputs['feats'], programs_pred)
        else:
            try:
                output = model(**inputs)
            except:
                output_ = model(image=inputs['feats'], question=inputs['questions'])
                output, _, _ = output_
        print(idx_to_answer_token[torch.argmax(output).cpu().item() + 4])
        if target is None:
            output.backward(output.detach())
        else:
            output.backward(target)

        grad = hook.gradient.float()
        act = hook.activation.float()

        alpha = torch.max(grad, dim=1, keepdim=True)[0]
        gradcam = torch.sum(act * alpha, dim=1, keepdim=True)
        gradcam = (gradcam - gradcam.mean()) / gradcam.std()
        # We only want neurons with positive influence so we
        # clamp any negative ones.
        gradcam = torch.clamp(gradcam, min=0)

    gradcam = F.interpolate(
        gradcam,
        torch.Size([224, 224]),
        mode='bicubic',
        align_corners=False).squeeze().detach().cpu().numpy()
    gradcam = np.pad(gradcam[30:194, 30:194], 30, pad_with, padder=0) + np.zeros((224, 224))
    return [gradcam]


def attCAM(
        model,
        inputs, model_type='monolith'):
    if model_type == 'monlith':
        output = model(**inputs)
        print(idx_to_answer_token[torch.argmax(output).cpu().item()])
        att_maps = model.att_maps
        final_maps = []
        for map in att_maps:
            range_ = (map.max() - map.min())
            map = map - map.min() / range_
            final_maps.append(F.interpolate(
                map,
                torch.Size([224, 224]),
                mode='bicubic',
                align_corners=False).squeeze().detach().cpu().numpy())
    elif model_type == 'generator_engine':
        programs_pred = model[0](inputs['questions'])
        scores = model[1](inputs['feats'], programs_pred, save_activations=True)
        att_maps = model[1].att_maps
        print(idx_to_answer_token[torch.argmax(scores).cpu().item()])
        att_maps = [f.mean(dim=1, keepdim=True) for f in att_maps]
        final_maps = []
        for map in att_maps:
            std = map.std()
            map = map - map.mean() / std
            map = torch.clamp(map, min=0)
            final_maps.append(F.interpolate(
                map,
                torch.Size([224, 224]),
                mode='bicubic',
                align_corners=False).squeeze().detach().cpu().numpy())
    elif model_type == 'generator_engine_sample':
        files = os.listdir('C:\\Users\\Guldan\\Desktop')
        mfd = []
        for file in files:
            if f'act_' in file:
                mfd.append(f'C:\\Users\\Guldan\\Desktop\{file}')

        for i in range(len(mfd)):
            os.remove(mfd[i])

        programs_pred = model[0].reinforce_sample(
            inputs['questions'],
            temperature=1,
            argmax=True)
        scores = model[1](inputs['feats'], programs_pred)
        print(idx_to_answer_token[torch.argmax(scores).cpu().item()])
        files = os.listdir('C:\\Users\\Guldan\\Desktop')
        att_maps = []
        for file in files:
            if f'act_' in file:
                with open(f'C:\\Users\\Guldan\\Desktop\{file}', 'rb') as f:
                    att_maps.append(torch.FloatTensor(np.load(f)))
        att_maps = [f.mean(dim=1, keepdim=True) for f in att_maps]
        final_maps = []
        for map in att_maps:
            std = map.std()
            map = (map - map.mean()) / std
            map = torch.clamp(map, min=0)
            final_maps.append(F.interpolate(
                map,
                torch.Size([224, 224]),
                mode='bicubic',
                align_corners=False).squeeze().detach().cpu().numpy())
    return final_maps


device = "cuda" if torch.cuda.is_available() else "cpu"
###---- Choose Model ----###
model = 'RNFP'
if model == 'SA':
    model_type = 'monolith'
    resnet = load_sa_resnet()
    model = load_cnn_sa()
    model.train()
    model.to(device)
    layer = "stacked-attn-1"
elif model == 'FiLM':
    model_type = 'generator_engine'
    resnet = load_film_resnet()
    program_generator, execution_engine = load_film()
    program_generator.eval()
    execution_engine.eval()
    program_generator.to(device)
    execution_engine.to(device)
    model = program_generator, execution_engine
    layer = getattr(model[1], '3')
elif model == 'IEP':
    model_type = 'generator_engine_sample'
    resnet = load_iep_resnet()
    program_generator, execution_engine = load_iep()
    program_generator.eval()
    execution_engine.eval()
    program_generator.to(device)
    execution_engine.to(device)
    model = program_generator, execution_engine
elif model == 'RNFP':
    model_type = 'monolith'
    resnet = None
    model = load_rnfp()
    model.train()
    model.to(device)
    layer = getattr(model.cn, 'conv4')


image_name = '../data/images/val/CLEVR_val_000000.png'

###---- Image Preprocess ----####
image_input = imread(image_name)
try:
    image_input = rgba2rgb(image_input)
except:
    pass
img = imresize(image_input, (224, 224))
image_np = copy.deepcopy(img)
img = img.astype('float32')
img = img.transpose(2, 0, 1)[None]
mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
std = np.array([0.229, 0.224, 0.224]).reshape(1, 3, 1, 1)
img = (img - mean) / std
img_var = torch.FloatTensor(img).to('cuda')
if resnet is not None:
    image_tensor = resnet(img_var)
else:
    image_tensor = img_var
###---- Image Preprocess ----####


print("Enter your question...\n")
# input_question = input().strip().lower()
input_question = 'Is there a purple object'
tokenized_question = torch.LongTensor(
    [1] + [question_token_to_idx[f] for f in input_question.split(' ')] + [2]).unsqueeze(0).to('cuda')

print("Enter expected answer...\n")
# input_answer = input().strip().lower()
input_answer = 'yes'
tokenized_answer = torch.LongTensor(one_hot(answer_token_to_idx[input_answer], 32)).unsqueeze(0).to('cuda')

# att_maps = attCAM(
#     model=model,
#     inputs={'questions': tokenized_question, 'feats': image_tensor}, model_type=model_type)



att_maps = gradCAM(
    model=model,
    inputs={'questions': tokenized_question, 'feats': image_tensor}, target=None,
    layer=layer)

for att_map in att_maps:
    viz_attn(image_np, att_map, True)
