import json
import os

import numpy as np
import torch
import torchvision
from torch.autograd import Variable

from fool_models.stack_attention import CnnLstmSaModel
from neural_render.blender_render_utils.constants import find_platform_slash
from utils.train_utils import ImageCLEVR_HDF5
from skimage.color import rgba2rgb
from skimage.io import imread
from skimage.transform import resize as imresize
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


def load_cnn_sa(baseline_model=f'{UP_TO_HERE_}/fool_models/resources/cnn_lstm_sa_mlp.pt'):
    model, _ = CnnLstmSaModel.load(baseline_model)
    model.eval()
    return model


def load_loader():
    val_set = ImageCLEVR_HDF5(config=None, split='val',
                              clvr_path='C:\\Users\\Guldan\\Desktop\\DeltaFormers\\data',
                              questions_path='C:\\Users\\Guldan\\Desktop\\DeltaFormers\\data',
                              scenes_path='C:\\Users\\Guldan\\Desktop\\DeltaFormers\\data',
                              use_cache=False,
                              return_program=False,
                              effective_range=10, output_shape=224)

    val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=1,
                                                 num_workers=0, shuffle=False, drop_last=False)
    return val_dataloader


def load_resnet_backbone():
    whole_cnn = getattr(torchvision.models, 'resnet101')(pretrained=True)
    layers = [
        whole_cnn.conv1,
        whole_cnn.bn1,
        whole_cnn.relu,
        whole_cnn.maxpool,
    ]
    for i in range(3):
        name = 'layer%d' % (i + 1)
        layers.append(getattr(whole_cnn, name))
    cnn = torch.nn.Sequential(*layers)
    cnn.type(torch.cuda.FloatTensor)
    cnn.eval()
    return cnn


def inference_with_cnn_sa(loader=None, model=None, resnet_extractor=None):
    dtype = torch.cuda.FloatTensor
    model.type(dtype)
    model.eval()
    num_correct, num_samples = 0, 0
    final_preds = []
    print(f"Testing for {len(loader)} samples")
    print()
    for batch in loader:
        (_, iq), answer, _ = batch
        # iq, answers = batch
        image = iq['image'].to('cuda')
        questions = iq['question'].to('cuda')
        feats = resnet_extractor(image)

        questions_var = Variable(questions.type(dtype).long())
        feats_var = Variable(feats.type(dtype))
        scores = model(questions_var, feats_var)

        _, preds = scores.data.cpu().max(1)
        for item in preds.detach().cpu().numpy():
            final_preds.append(item - 4)
        num_correct += (preds == (answer.squeeze() + 4)).sum()
        num_samples += preds.size(0)
        if num_samples % 1000 == 0:
            print(f'Ran {num_samples} samples at {float(num_correct) / num_samples} accuracy')

    acc = float(num_correct) / num_samples
    print('Got %d / %d = %.2f correct' % (num_correct, num_samples, 100 * acc))
    return final_preds


def single_inference_with_cnn_sa(model=None, resnet=None):
    dtype = torch.cuda.FloatTensor
    model.type(dtype)
    model.eval()
    img_size = (224, 224)
    path = '../neural_render/images'
    images = [f'../neural_render/images/{f}' for f in os.listdir(path) if 'Rendered' in f and '.png' in f]
    feat_list = []
    ### Read the images ###
    for image in images:
        img = imread(image)
        img = rgba2rgb(img)
        img = imresize(img, img_size)
        img = img.astype('float32')
        img = img.transpose(2, 0, 1)[None]
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        std = np.array([0.229, 0.224, 0.224]).reshape(1, 3, 1, 1)
        img = (img - mean) / std
        img_var = torch.FloatTensor(img).to('cuda')
        ### Pass through Resnet ###
        feat_list.append(resnet(img_var))

    ### Stack them with Questions ###
    feats = torch.cat(feat_list, dim=0)
    questions = [10, 85, 14, 25, 30, 64, 66, 84, 74, 75, 21, 84, 45, 86, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    questions = torch.LongTensor(questions).unsqueeze(0)

    questions = questions.to('cuda')
    feats = feats.to('cuda')
    scores = model(questions, feats)
    _, preds = scores.data.cpu().max(1)
    preds = [f - 4 for f in preds]
    print(preds)
    return


#resnet = load_resnet_backbone()
#model = load_cnn_sa()
# loader = load_loader()
# inference_with_cnn_sa(loader=loader, model=model, resnet_extractor=resnet)
#single_inference_with_cnn_sa(model=model, resnet=resnet)
