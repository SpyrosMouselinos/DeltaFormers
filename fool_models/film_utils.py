import inspect
import json
import os
import numpy as np
import torch
import torchvision
from torch.autograd import Variable

from fool_models.film import FiLMGen, FiLMedNet
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

arg_value_updates = {
    'condition_method': {
        'block-input-fac': 'block-input-film',
        'block-output-fac': 'block-output-film',
        'cbn': 'bn-film',
        'conv-fac': 'conv-film',
        'relu-fac': 'relu-film',
    },
    'module_input_proj': {
        True: 1,
    },
}


def get_updated_args(kwargs, object_class):
    """
    Returns kwargs with renamed args or arg valuesand deleted, deprecated, unused args.
    Useful for loading older, trained models.
    If using this function is neccessary, use immediately before initializing object.
    """
    # Update arg values
    for arg in arg_value_updates:
        if arg in kwargs and kwargs[arg] in arg_value_updates[arg]:
            kwargs[arg] = arg_value_updates[arg][kwargs[arg]]

    # Delete deprecated, unused args
    valid_args = inspect.getfullargspec(object_class.__init__)[0]
    new_kwargs = {valid_arg: kwargs[valid_arg] for valid_arg in valid_args if valid_arg in kwargs}
    return new_kwargs


def load_cpu(path):
    """
    Loads a torch checkpoint, remapping all Tensors to CPU
    """
    return torch.load(path, map_location=lambda storage, loc: storage)


def load_program_generator(path):
    checkpoint = load_cpu(path)
    kwargs = checkpoint['program_generator_kwargs']
    state = checkpoint['program_generator_state']
    kwargs = get_updated_args(kwargs, FiLMGen)
    model = FiLMGen(**kwargs)
    model.load_state_dict(state)
    return model, kwargs


def load_execution_engine(path, verbose=False):
    checkpoint = load_cpu(path)
    kwargs = checkpoint['execution_engine_kwargs']
    state = checkpoint['execution_engine_state']
    kwargs['verbose'] = verbose
    kwargs = get_updated_args(kwargs, FiLMedNet)
    model = FiLMedNet(**kwargs)
    model.load_state_dict(state)
    return model, kwargs


def load_film(program_generator=f'{UP_TO_HERE_}/fool_models/resources/film.pt',
              execution_engine=f'{UP_TO_HERE_}/fool_models/resources/film.pt'):
    program_generator, _ = load_program_generator(program_generator)
    program_generator.train()
    execution_engine, _ = load_execution_engine(execution_engine, verbose=False)
    execution_engine.train()
    model = (program_generator, execution_engine)
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


def inference_with_film(loader=None, model=None, resnet_extractor=None):
    dtype = torch.cuda.FloatTensor
    program_generator, execution_engine = model
    program_generator.type(dtype)
    program_generator.eval()
    execution_engine.type(dtype)
    execution_engine.eval()

    num_correct, num_samples = 0, 0
    final_preds = []
    print(f"Testing for {len(loader)} samples")
    print()
    for batch in loader:
        (_, iq), answer, _ = batch
        image = iq['image'].to('cuda')
        questions = iq['question'].to('cuda')
        feats = resnet_extractor(image)

        questions_var = Variable(questions.type(dtype).long())
        feats_var = Variable(feats.type(dtype))

        programs_pred = program_generator(questions_var)
        scores = execution_engine(feats_var, programs_pred, save_activations=False)

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

def sinference_with_film(model=None, loader=None, resnet_extractor=None):
    folder = 'C:\\Users\\Guldan\\Desktop\\DeltaFormers\\results\\images\\film'
    images = [f for f in os.listdir(folder) if '.png' in f]
    texts = [f for f in os.listdir(folder) if '.txt' in f]
    matches = []
    for image in images:
        for text in texts:
            if image.split('_')[0] == text.split('_')[0]:
                with open(folder + '\\' +text, 'r') as fin:
                    question = fin.readline()
                    answer = fin.readline().strip().split(' ')[-1]
                matches.append((image,question, answer))

    for blob in matches:
        image_path, question, answer = blob
        dtype = torch.cuda.FloatTensor
        program_generator, execution_engine = model
        program_generator.type(dtype)
        program_generator.eval()
        execution_engine.type(dtype)
        execution_engine.eval()
        img = imread(folder + '\\' + image_path)
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

        programs_pred = program_generator(questions_var)
        scores = execution_engine(image, programs_pred, save_activations=False)
        _, preds = scores.data.cpu().max(1)
        final_preds = []
        for item in preds.detach().cpu().numpy():
            final_preds.append(item - 4)

        if answer != idx_to_answer_token[final_preds[0] + 4]:
            print(image_path)
            print(answer)
            print(idx_to_answer_token[final_preds[0] + 4])
            print(idx_to_answer_token[final_preds[0]])
            print("--------------")
    return

#resnet = load_resnet_backbone()
#model = load_film()
# loader = load_loader()
# inference_with_film(loader=loader, model=model, resnet_extractor=resnet)
#sinference_with_film(model,None,resnet)
