import os
import os.path as osp
import random
import sys
from shutil import copyfile

import yaml

sys.path.insert(0, osp.abspath('.'))

import numpy as np
import argparse
from torch.utils.data import Dataset
from modules.embedder import *
from utils.train_utils import StateCLEVR, ImageCLEVR, ImageCLEVR_HDF5
from fool_models.film_utils import load_film, load_resnet_backbone


def _print(something):
    print(something, flush=True)
    return


AVAILABLE_DATASETS = {
    'DeltaRN': [StateCLEVR],
    'DeltaSQFormer': [StateCLEVR],
    'DeltaQFormer': [StateCLEVR],
    'DeltaSQFormerCross': [StateCLEVR],
    'DeltaSQFormerDisentangled': [StateCLEVR],
    'DeltaSQFormerLinear': [StateCLEVR],
    'DeltaSQFormerPreLNLinear': [StateCLEVR],
    'DeltaRNFP': [ImageCLEVR, ImageCLEVR_HDF5],
}

AVAILABLE_MODELS = {'DeltaRN': DeltaRN,
                    'DeltaRNFP': DeltaRNFP,
                    'DeltaSQFormer': DeltaSQFormer,
                    'DeltaSQFormerCross': DeltaSQFormerCross,
                    'DeltaSQFormerDisentangled': DeltaSQFormerDisentangled,
                    'DeltaSQFormerLinear': DeltaSQFormerLinear,
                    'DeltaSQFormerPreLNLinear': DeltaSQFormerPreLNLinear,
                    'DeltaQFormer': DeltaQFormer}


def save_all(model, optim, sched, bs_sched, epoch, loss, path, train_percentage):
    torch.save({
        'epoch': epoch,
        'val_loss': loss,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        'scheduler_state_dict': sched.state_dict(),
        'bs_scheduler_state_dict': bs_sched.state_dict(),
    }, path + f'/model_at_{train_percentage}.pt')
    return


def save_some(model, path, train_percentage):
    torch.save({
        'model_state_dict': model.state_dict(),
    }, path + f'/model_at_{train_percentage}.pt')
    return


def load(path: str, model: nn.Module, optim=None, sched=None, bs_sched=None, mode='all'):
    _print("Remember that available modes are: [all, model, model+opt, model+opt+sched, model+opt+sched+bs_sched]\n")
    checkpoint = torch.load(path)
    # removes 'module' from dict entries, pytorch bug #3805
    if torch.cuda.device_count() >= 1 and any(k.startswith('module.') for k in checkpoint['model_state_dict'].keys()):
        checkpoint['model_state_dict'] = {k.replace('module.', ''): v for k, v in
                                          checkpoint['model_state_dict'].items()}
    # if torch.cuda.device_count() > 1 and not any(k.startswith('module.') for k in checkpoint.keys()):
    #     checkpoint = {'module.' + k: v for k, v in checkpoint.items()}
    epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model_state_dict'])
    if ('opt' in mode or 'all' in mode) and optim is not None:
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        _print("Optimizer not Loaded!\n")

    if ('sched' in mode or 'all' in mode) and optim is not None:
        sched.load_state_dict(checkpoint['scheduler_state_dict'])
    else:
        _print("Scheduler not Loaded!\n")

    if ('bs_sched' in mode or 'all' in mode) and optim is not None:
        bs_sched.load_state_dict(checkpoint['bs_scheduler_state_dict'])
    else:
        _print("BS Scheduler not Loaded!\n")
    _print(f"Your model achieves {round(checkpoint['val_loss'], 4)} validation loss\n")
    return model, optim, sched, bs_sched, epoch


def kwarg_dict_to_device(data_obj, device):
    if device == 'cpu':
        return data_obj
    cpy = {}
    for key, _ in data_obj.items():
        cpy[key] = data_obj[key].to(device)
    return cpy


def check_paths(config, experiment_name):
    if osp.exists('results'):
        pass
    else:
        os.mkdir(f'results/')

    if osp.exists(f'results/{experiment_name}'):
        pass
    else:
        os.mkdir(f'results/{experiment_name}')

    copyfile(config, f'./results/{experiment_name}/config.yaml')
    return


def accuracy_metric(y_pred, y_true):
    acc = (y_pred.argmax(1) == y_true).float().detach().cpu().numpy()
    return float(100 * acc.sum() / len(acc))


def train_model(device, experiment_name='experiment_1', clvr_path='data/',
                questions_path='data/', scenes_path='data/',
                train_percentage=20, random_seed=0):
    effective_range_percentage = train_percentage / 100
    random.seed(random_seed)
    np.random.seed(random_seed)

    if device == 'cuda':
        device = 'cuda:0'

    experiment_name = experiment_name
    config = f'./results/{experiment_name}/config_dfp.yaml'
    with open(config, 'r') as fin:
        config = yaml.load(fin, Loader=yaml.FullLoader)

    # experiment_name = load_from.split('results/')[-1].split('/')[0]
    # config = f'./results/{experiment_name}/config.yaml'
    # with open(config, 'r') as fin:
    #     config = yaml.load(fin, Loader=yaml.FullLoader)

    # model = AVAILABLE_MODELS[config['model_architecture']](config)
    # _print(f"Loading Model of type: {config['model_architecture']}\n")
    # model, _, _, _, _ = load(path=load_from, model=model, mode='model')

    resnet = load_resnet_backbone()
    resnet.eval()
    program_generator, execution_engine = load_film()
    program_generator = program_generator.to(device)
    execution_engine = execution_engine.to(device)
    program_generator.train()
    execution_engine.train()

    train_set = AVAILABLE_DATASETS[config['model_architecture']][1](config=config, split='Limits',
                                                                    clvr_path=clvr_path,
                                                                    questions_path=questions_path,
                                                                    scenes_path=scenes_path,
                                                                    use_cache=False, return_program=False,
                                                                    effective_range_offset=0,
                                                                    randomize_range=False,
                                                                    effective_range=effective_range_percentage,
                                                                    prior_shuffle=False, output_shape=224)

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=2, shuffle=False)

    val_set = AVAILABLE_DATASETS[config['model_architecture']][1](config=config, split='Limits',
                                                                  clvr_path=clvr_path,
                                                                  questions_path=questions_path,
                                                                  scenes_path=scenes_path, use_cache=False,
                                                                  return_program=False,
                                                                  effective_range_offset=effective_range_percentage,
                                                                  randomize_range=False, effective_range=None,
                                                                  prior_shuffle=False,
                                                                  output_shape=224)

    val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False)

    optimizer1 = torch.optim.AdamW(params=program_generator.parameters(), lr=1e-4, weight_decay=2e-5)
    optimizer2 = torch.optim.AdamW(params=execution_engine.parameters(), lr=1e-4, weight_decay=2e-5)

    init_epoch = 0

    criterion = nn.CrossEntropyLoss()
    metric = accuracy_metric

    total_loss = 0.
    total_acc = 0.
    best_val_acc = -1
    overfit_count = -3
    optimizer1.zero_grad()
    optimizer2.zero_grad()

    flag = False
    running_train_batch_index = 0

    for epoch in range(init_epoch, 20):
        _print(f"Epoch: {epoch}\n")
        if flag:
            _print(
                f"Validating at Epoch: {epoch} and Total Batch Index {running_train_batch_index}\n")
            total_val_loss = 0.
            total_val_acc = 0.
            # Turn off the train mode #
            program_generator.eval()
            execution_engine.eval()
            with torch.no_grad():
                for val_batch_index, val_batch in enumerate(val_dataloader):
                    data, y_real = val_batch
                    data = kwarg_dict_to_device(data, device)
                    y_real = y_real.to(device)
                    feats = resnet(data['image'])
                    programs = program_generator(data['question'])
                    y_pred = execution_engine(feats, programs)
                    val_loss = criterion(y_pred, y_real.squeeze(1))
                    val_acc = metric(y_pred, y_real.squeeze(1))
                    total_val_loss += val_loss.item()
                    total_val_acc += val_acc
            _print('| epoch {:3d} | val loss {:5.2f} | val acc {:5.2f} \n'.format(epoch,
                                                                                  total_val_loss / (
                                                                                          val_batch_index + 1),
                                                                                  total_val_acc / (
                                                                                          val_batch_index + 1)))
            if total_val_acc / (val_batch_index + 1) > best_val_acc:
                best_val_acc = total_val_acc / (val_batch_index + 1)
            print(f"Final Results for {train_percentage} Percentage!\n")
            print(best_val_acc)
            return
        for train_batch_index, train_batch in enumerate(train_dataloader):
            if running_train_batch_index % 500 == 0 and running_train_batch_index > 0:
                _print(
                    f"Validating at Epoch: {epoch} and Total Batch Index {running_train_batch_index}\n")
                total_val_loss = 0.
                total_val_acc = 0.
                # Turn off the train mode #
                execution_engine.eval()
                program_generator.eval()
                with torch.no_grad():
                    for val_batch_index, val_batch in enumerate(val_dataloader):
                        data, y_real = val_batch
                        data = kwarg_dict_to_device(data, device)
                        y_real = y_real.to(device)
                        feats = resnet(data['image'])
                        programs = program_generator(data['question'])
                        y_pred = execution_engine(feats, programs)
                        val_loss = criterion(y_pred, y_real.squeeze(1))
                        val_acc = metric(y_pred, y_real.squeeze(1))
                        total_val_loss += val_loss.item()
                        total_val_acc += val_acc
                _print('| epoch {:3d} | val loss {:5.2f} | val acc {:5.2f} \n'.format(epoch,
                                                                                      total_val_loss / (
                                                                                              val_batch_index + 1),
                                                                                      total_val_acc / (
                                                                                              val_batch_index + 1)))
                if total_val_acc / (val_batch_index + 1) > best_val_acc:
                    best_val_acc = total_val_acc / (val_batch_index + 1)
                    if best_val_acc > 99.5:
                        flag = True
                    overfit_count = -1
                else:
                    overfit_count += 1
                    if overfit_count % config['early_stopping'] == 0 and overfit_count > 0:
                        _print(f"Training stopped at epoch: {epoch} and best validation acc: {best_val_acc}")
                        return
                program_generator.train()
                execution_engine.train()

            else:
                # Turn on the train mode #
                data, y_real = train_batch
                data = kwarg_dict_to_device(data, device)
                y_real = y_real.to(device)
                feats = resnet(data['image'])
                programs = program_generator(data['question'])
                y_pred = execution_engine(feats, programs)
                loss = criterion(y_pred, y_real.squeeze(1))
                acc = metric(y_pred, y_real.squeeze(1))
                loss.backward()
                optimizer1.step()
                optimizer2.step()
                optimizer1.zero_grad()
                optimizer2.zero_grad()
                total_loss += loss.item()
                total_acc += acc
                if train_batch_index % int(len(train_dataloader) / 2) == 0 and train_batch_index > 0:
                    cur_loss = total_loss / (train_batch_index + 1)
                    cur_acc = total_acc / (train_batch_index + 1)
                    _print('| epoch {:3d} | {:5d}/{:5d} batches | '
                           'lr {:02.6f} | loss {:5.2f} | acc {:5.2f}\n'.format(epoch, train_batch_index,
                                                                               len(train_dataloader),
                                                                               1e-5, cur_loss,
                                                                               cur_acc))

                    if cur_acc >= 99:
                        flag = True
            # End of batch #
            running_train_batch_index += 1
        # End of epoch #
        total_loss = 0.
        total_acc = 0.

    with open(f'C:\\Users\\Spyros\\Desktop\\generalize\\film_percentage_{train_percentage}_{random_seed}.txt',
              'w') as fout:
        fout.write('GAME\n')
        fout.write(str(best_val_acc))

    print(f"Final Results for {train_percentage} Percentage!\n")
    print(best_val_acc)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, help='The name of the experiment', default='experiment_dfp')
    parser.add_argument('--config', type=str, help='The path to the config file', default='./config_dfp.yaml')
    parser.add_argument('--device', type=str, help='cpu or cuda', default='cuda')

    parser.add_argument('--load_from', type=str, help='continue training',
                        default='None')
    parser.add_argument('--scenes_path', type=str, help='folder of scenes', default='data/')
    parser.add_argument('--questions_path', type=str, help='folder of questions', default='data/')
    parser.add_argument('--clvr_path', type=str, help='folder before images', default='data/')
    parser.add_argument('--use_cache', type=int, help='if to use cache (only in image clever)', default=0)
    parser.add_argument('--use_hdf5', type=int, help='if to use hdf5 loader', default=1)
    parser.add_argument('--freeze_exponential_growth', type=int, help='if to stay on same lr uppon resume', default=0)
    parser.add_argument('--train_percentage', type=float,
                        help='The percentage of Defense Dataset to be used as training', default=10)
    args = parser.parse_args()

    if args.freeze_exponential_growth == 0:
        args.freeze_exponential_growth = False
    else:
        args.freeze_exponential_growth = True

    if args.use_cache == 0:
        args.use_cache = False
    else:
        args.use_cache = True

    if args.use_hdf5 == 0:
        args.use_hdf5 = False
    else:
        args.use_hdf5 = True

    for train_percentage in [5, 10, 15, 20, 25, 30]:
        train_model(device=args.device, experiment_name=args.name, clvr_path=args.clvr_path,
                    questions_path=args.questions_path, scenes_path=args.scenes_path, train_percentage=train_percentage,
                    random_seed=train_percentage + 0)
