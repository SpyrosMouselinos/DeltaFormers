import json
import os

import torch
import torchvision
from torch.autograd import Variable

from fool_models.iep import Seq2Seq, ModuleNet
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


def load_cpu(path):
    """
    Loads a torch checkpoint, remapping all Tensors to CPU
    """
    return torch.load(path, map_location=lambda storage, loc: storage)


def load_program_generator(path):
    checkpoint = load_cpu(path)
    kwargs = checkpoint['program_generator_kwargs']
    state = checkpoint['program_generator_state']
    model = Seq2Seq(**kwargs)
    #model.load_state_dict(state)
    return model, kwargs


def load_execution_engine(path, verbose=True):
    checkpoint = load_cpu(path)
    kwargs = checkpoint['execution_engine_kwargs']
    state = checkpoint['execution_engine_state']
    kwargs['verbose'] = verbose
    model = ModuleNet(**kwargs)
    #model.load_state_dict(state)
    return model, kwargs


def load_iep(program_generator=f'{UP_TO_HERE_}/fool_models/resources/program_generator_700k.pt',
             execution_engine=f'{UP_TO_HERE_}/fool_models/resources/execution_engine_700k_strong.pt'):
    program_generator, _ = load_program_generator(program_generator)
    #program_generator.eval()
    execution_engine, _ = load_execution_engine(execution_engine, verbose=False)
    #execution_engine.eval()
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


def inference_with_iep(loader=None, model=None, resnet_extractor=None):
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
        (_, iq), answer, _ = batch
        image = iq['image'].to('cuda')
        questions = iq['question'].to('cuda')
        feats = resnet_extractor(image)

        questions_var = Variable(questions.type(dtype).long())
        feats_var = Variable(feats.type(dtype))

        programs_pred = program_generator.reinforce_sample(
            questions_var,
            temperature=1,
            argmax=True)

        scores = execution_engine(feats_var, programs_pred)

        _, preds = scores.data.cpu().max(1)
        for item in preds.detach().cpu().numpy():
            final_preds.append(item - 4)
        num_correct += (preds == (answer.squeeze() + 4)).sum()
        num_samples += preds.size(0)
        if num_samples % 1000 == 0:
            print(f'Ran {num_samples} samples at {float(num_correct) / num_samples} accuracy')

    acc = float(num_correct) / num_samples
    print('[IEP] Got %d / %d = %.2f correct' % (num_correct, num_samples, 100 * acc))
    return final_preds


# resnet = load_resnet_backbone()
# model = load_iep()
# loader = load_loader()
# inference_with_iep(loader=loader, model=model, resnet_extractor=resnet)
