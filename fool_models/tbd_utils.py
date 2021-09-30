import json
import os

import numpy as np
import torch
import torchvision
from skimage.color import rgba2rgb
from skimage.io import imread
from skimage.transform import resize as imresize

from fool_models.tbd import _Seq2Seq, TbDNet
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
    # TBD Takes 2nd layer features #
    whole_cnn = getattr(torchvision.models, 'resnet101')(pretrained=True)
    layers = [
        whole_cnn.conv1,
        whole_cnn.bn1,
        whole_cnn.relu,
        whole_cnn.maxpool,
    ]
    for i in range(2):
        name = 'layer%d' % (i + 1)
        layers.append(getattr(whole_cnn, name))
    cnn = torch.nn.Sequential(*layers)
    cnn.type(torch.cuda.FloatTensor)
    cnn.eval()
    return cnn


def load_program_generator(checkpoint=f'{UP_TO_HERE_}/fool_models/resources/program_generator_tbd.pt'):
    checkpoint = torch.load(str(checkpoint), map_location={'cuda:0': 'cpu'})
    kwargs = checkpoint['program_generator_kwargs']
    state = checkpoint['program_generator_state']
    program_generator = _Seq2Seq(**kwargs)
    program_generator.load_state_dict(state)
    return program_generator


def load_tbd_net(checkpoint=f'{UP_TO_HERE_}/fool_models/resources/tbd.pt', vocab=data):
    tbd_net = TbDNet(vocab)
    tbd_net.load_state_dict(torch.load(str(checkpoint), map_location={'cuda:0': 'cpu'}))
    if torch.cuda.is_available():
        tbd_net.cuda()
    return tbd_net


def load_tbd():
    program_generator = load_program_generator()
    program_generator.cuda()
    program_generator.eval()
    execution_engine = load_tbd_net()
    execution_engine.cuda()
    execution_engine.eval()
    model = (program_generator, execution_engine)
    return model


def inference_with_tbh(loader=None, model=None, resnet_extractor=None):
    final_preds = []
    dtype = torch.cuda.FloatTensor
    program_generator, execution_engine = model
    program_generator.type(dtype)
    program_generator.eval()
    execution_engine.type(dtype)
    execution_engine.eval()

    num_correct, num_samples = 0, 0
    print(f"Testing for {len(loader)} samples")
    print()
    for batch in loader:
        (_, iq), answers, _ = batch
        # iq, answers = batch
        image = iq['image'].to('cuda')
        questions_var = iq['question'].to('cuda')
        feats_var = resnet_extractor(image)

        progs = []
        for i in range(questions_var.size(0)):
            program = program_generator.reinforce_sample(questions_var[i, :].view(1, -1))
            progs.append(program.cpu().numpy().squeeze())
        progs = np.asarray(progs)

        scores = execution_engine(feats_var, torch.LongTensor(progs))

        _, preds = scores.data.cpu().max(1)

        correct_preds = []
        for item in preds.detach().cpu().numpy():
            correct_preds.append(execution_engine.translate_codes[item] - 4)

        for item in correct_preds:
            # For line 684 in reinforce_train.py
            final_preds.append(item)

        num_correct += (torch.LongTensor(correct_preds) == (answers.squeeze())).sum()
        num_samples += preds.size(0)
        if num_samples % 1000 == 0:
            print(f'Ran {num_samples} samples at {float(num_correct) / num_samples} accuracy')

    acc = float(num_correct) / num_samples
    print('[TBD] Got %d / %d = %.2f correct' % (num_correct, num_samples, 100 * acc))
    return final_preds


def sinference_with_tbh(model=None, resnet_extractor=None):
    final_preds = []
    dtype = torch.cuda.FloatTensor
    program_generator, execution_engine = model
    program_generator.type(dtype)
    program_generator.eval()
    execution_engine.type(dtype)
    execution_engine.eval()
    img = imread('C:\\Users\\Guldan\\Desktop\\saveme3.png')
    question = 'Is the shape of the brown rubber object the same as the thing that is left of the big object ;'
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
    image = resnet_extractor(img_var)
    image = image.to('cuda')
    questions_var = torch.LongTensor([question_token_to_idx[f] for f in question[:-1].split(' ')]).to('cuda')
    questions_var = questions_var.unsqueeze(0)
    progs = []
    for i in range(questions_var.size(0)):
        program = program_generator.reinforce_sample(questions_var[i, :].view(1, -1))
        progs.append(program.cpu().numpy().squeeze())
    progs = np.asarray(progs)

    scores = execution_engine(image, torch.LongTensor(progs))

    _, preds = scores.data.cpu().max(1)

    correct_preds = []
    for item in preds.detach().cpu().numpy():
        correct_preds.append(execution_engine.translate_codes[item] - 4)

    print(correct_preds)
    print(idx_to_answer_token[correct_preds[0] + 4])

    return final_preds


#model = load_tbd()
# loader = load_loader()
#resnet_extractor = load_resnet_backbone()
#inference_with_tbh(loader=None, model=model, resnet_extractor=resnet_extractor)
#sinference_with_tbh(model=model, resnet_extractor=resnet_extractor)