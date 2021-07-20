import os
import os.path as osp
import sys
from shutil import copyfile

import yaml

sys.path.insert(0, osp.abspath('.'))

import argparse
from torch.utils.data import Dataset
from modules.embedder import *
from utils.train_utils import StateCLEVR, ImageCLEVR, ImageCLEVR_HDF5, BatchSizeScheduler


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
    'DeltaRNFP': [ImageCLEVR, ImageCLEVR_HDF5],
}

AVAILABLE_MODELS = {'DeltaRN': DeltaRN,
                    'DeltaRNFP': DeltaRNFP,
                    'DeltaSQFormer': DeltaSQFormer,
                    'DeltaSQFormerCross': DeltaSQFormerCross,
                    'DeltaSQFormerDisentangled': DeltaSQFormerDisentangled,
                    'DeltaSQFormerLinear': DeltaSQFormerLinear,
                    'DeltaQFormer': DeltaQFormer}


def save_all(model, optim, sched, bs_sched, epoch, loss, path):
    torch.save({
        'epoch': epoch,
        'val_loss': loss,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        'scheduler_state_dict': sched.state_dict(),
        'bs_scheduler_state_dict': bs_sched.state_dict(),
    }, path + f'/mos_epoch_{epoch}.pt')
    return


def load(path: str, model: nn.Module, optim=None, sched=None, bs_sched=None, mode='all'):
    _print("Remember that available modes are: [all, model, model+opt, model+opt+sched, model+opt+sched+bs_sched]\n")
    checkpoint = torch.load(path)
    # removes 'module' from dict entries, pytorch bug #3805
    if torch.cuda.device_count() >= 1 and any(k.startswith('module.') for k in checkpoint['model_state_dict'].keys()):
        checkpoint['model_state_dict'] = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()}
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


def train_model(config, device, experiment_name='experiment_1', load_from=None, clvr_path='data/',
                questions_path='data/', scenes_path='data/', use_cache=False, use_hdf5=False,
                freeze_exponential_growth=False):
    if device == 'cuda':
        device = 'cuda:0'

    if load_from is not None:
        experiment_name = load_from.split('results/')[-1].split('/')[0]
        config = f'./results/{experiment_name}/config.yaml'
        with open(config, 'r') as fin:
            config = yaml.load(fin, Loader=yaml.FullLoader)

        model = AVAILABLE_MODELS[config['model_architecture']](config)
        _print(f"Loading Model of type: {config['model_architecture']}\n")
        model = model.to(device)
        model.train()

        if use_hdf5:
            train_set = AVAILABLE_DATASETS[config['model_architecture']][1](config=config, split='train',
                                                                            clvr_path=clvr_path,
                                                                            questions_path=questions_path,
                                                                            scenes_path=scenes_path,
                                                                            use_cache=use_cache)
        else:
            # TODO: Change this!
            train_set = AVAILABLE_DATASETS[config['model_architecture']][0](config=config, split='train',
                                                                            clvr_path=clvr_path,
                                                                            questions_path=questions_path,
                                                                            scenes_path=scenes_path,
                                                                            use_cache=use_cache)
        train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=config['batch_size'],
                                                       num_workers=config['n_workers'], shuffle=True)

        if use_hdf5:
            val_set = AVAILABLE_DATASETS[config['model_architecture']][1](config=config, split='val',
                                                                          clvr_path=clvr_path,
                                                                          questions_path=questions_path,
                                                                          scenes_path=scenes_path, use_cache=use_cache)
        else:
            pass
            val_set = AVAILABLE_DATASETS[config['model_architecture']][0](config=config, split='val',
                                                                          clvr_path=clvr_path,
                                                                          questions_path=questions_path,
                                                                          scenes_path=scenes_path, use_cache=use_cache)

        val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=config['batch_size'],
                                                     num_workers=config['n_workers'], shuffle=False)

        optimizer = torch.optim.Adam(params=model.parameters(), lr=config['lr'], weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['scheduler_step_size'],
                                                    gamma=config['scheduler_gamma'])

        bs_scheduler = BatchSizeScheduler(train_ds=train_set, initial_bs=config['batch_size'],
                                          step_size=config['bs_scheduler_step_size'],
                                          gamma=config['bs_scheduler_gamma'],
                                          max_bs=config['max_batch_size'])
        model, optimizer, scheduler, bs_scheduler, init_epoch = load(path=load_from, model=model, optim=optimizer,
                                                                     sched=scheduler, bs_sched=bs_scheduler,
                                                                     mode='all')
        if freeze_exponential_growth:
            bs_scheduler.step_size = -1

    else:
        check_paths(config, experiment_name)
        with open(config, 'r') as fin:
            config = yaml.load(fin, Loader=yaml.FullLoader)

        if use_hdf5:
            train_set = AVAILABLE_DATASETS[config['model_architecture']][1](config=config, split='train',
                                                                            clvr_path=clvr_path,
                                                                            questions_path=questions_path,
                                                                            scenes_path=scenes_path,
                                                                            use_cache=use_cache)
        else:
            train_set = AVAILABLE_DATASETS[config['model_architecture']][0](config=config, split='train',
                                                                            clvr_path=clvr_path,
                                                                            questions_path=questions_path,
                                                                            scenes_path=scenes_path,
                                                                            use_cache=use_cache)

        train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=config['batch_size'],
                                                       num_workers=config['n_workers'], shuffle=True)
        _print(f"Loaded Train Dataset at {len(train_dataloader)} batches of size {config['batch_size']}")
        if use_hdf5:
            val_set = AVAILABLE_DATASETS[config['model_architecture']][1](config=config, split='val',
                                                                          clvr_path=clvr_path,
                                                                          questions_path=questions_path,
                                                                          scenes_path=scenes_path, use_cache=use_cache)
        else:
            val_set = AVAILABLE_DATASETS[config['model_architecture']][0](config=config, split='val',
                                                                          clvr_path=clvr_path,
                                                                          questions_path=questions_path,
                                                                          scenes_path=scenes_path, use_cache=use_cache)
        val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=config['batch_size'],
                                                     num_workers=config['n_workers'], shuffle=False)
        _print(f"Loaded Validation Dataset at {len(val_dataloader)} batches of size {config['batch_size']}")
        model = AVAILABLE_MODELS[config['model_architecture']](config)
        _print(f"Loading Model of type: {config['model_architecture']}\n")
        model = model.to(device)
        model.train()
        # TODO: Change this!
        optimizer = torch.optim.Adam(params=model.parameters(), lr=config['lr'], weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['scheduler_step_size'],
                                                    gamma=config['scheduler_gamma'])

        bs_scheduler = BatchSizeScheduler(train_ds=train_set, initial_bs=config['batch_size'],
                                          step_size=config['bs_scheduler_step_size'],
                                          gamma=config['bs_scheduler_gamma'],
                                          max_bs=config['max_batch_size'])
        init_epoch = 0

    ### Multi-GPU-Support ###
    if torch.cuda.device_count() > 1:
        _print(f"Let's use {torch.cuda.device_count()} GPUS")
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    metric = accuracy_metric

    total_loss = 0.
    total_acc = 0.
    best_val_loss = 1000
    overfit_count = -3
    log_interval = config['log_every']
    running_train_batch_index = 0
    for epoch in range(init_epoch, config['max_epochs']):
        _print(f"Epoch: {epoch}\n")
        for train_batch_index, train_batch in enumerate(train_dataloader):
            if running_train_batch_index % config['validate_every'] == 0 and running_train_batch_index > 0:
                _print(
                    f"Validating at Epoch: {epoch} and Total Batch Index {running_train_batch_index}\n")
                total_val_loss = 0.
                total_val_acc = 0.
                # Turn off the train mode #
                model.eval()
                with torch.no_grad():
                    for val_batch_index, val_batch in enumerate(val_dataloader):
                        data, y_real = val_batch
                        data = kwarg_dict_to_device(data, device)
                        y_real = y_real.to(device)
                        y_pred = model(**data)[0]
                        val_loss = criterion(y_pred, y_real.squeeze(1))
                        val_acc = metric(y_pred, y_real.squeeze(1))
                        total_val_loss += val_loss.item()
                        total_val_acc += val_acc
                _print('| epoch {:3d} | val loss {:5.2f} | val acc {:5.2f} \n'.format(epoch,
                                                                                      total_val_loss / (
                                                                                              val_batch_index + 1),
                                                                                      total_val_acc / (
                                                                                              val_batch_index + 1)))
                if total_val_loss / (val_batch_index + 1) < best_val_loss:
                    best_val_loss = total_val_loss / (val_batch_index + 1)

                    save_all(model, optimizer, scheduler, bs_scheduler, epoch, best_val_loss,
                             f'./results/{experiment_name}')

                    overfit_count = -1
                else:
                    overfit_count += 1
                    if overfit_count % config['early_stopping'] == 0 and overfit_count > 0:
                        _print(f"Training stopped at epoch: {epoch} and best validation loss: {best_val_loss}")
                        return
                model.train()

            else:
                # Turn on the train mode #
                data, y_real = train_batch
                data = kwarg_dict_to_device(data, device)
                y_real = y_real.to(device)
                optimizer.zero_grad()

                y_pred = model(**data)[0]
                loss = criterion(y_pred, y_real.squeeze(1))
                acc = metric(y_pred, y_real.squeeze(1))

                loss.backward()
                # Gradient Clipping

                if 'clip_grad_norm' not in config or config['clip_grad_norm'] == -1:
                    pass
                elif config['clip_grad_norm'] != -1:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad_norm'])

                optimizer.step()

                total_loss += loss.item()
                total_acc += acc
                if train_batch_index % log_interval == 0 and train_batch_index > 0:
                    cur_loss = total_loss / (train_batch_index + 1)
                    cur_acc = total_acc / (train_batch_index + 1)
                    _print('| epoch {:3d} | {:5d}/{:5d} batches | '
                           'lr {:02.6f} | loss {:5.2f} | acc {:5.2f}\n'.format(epoch, train_batch_index,
                                                                               len(train_dataloader),
                                                                               scheduler.get_last_lr()[0], cur_loss,
                                                                               cur_acc))
            # End of batch #
            running_train_batch_index += 1
        # End of epoch #
        total_loss = 0.
        total_acc = 0.
        if scheduler.get_last_lr()[0] < config['max_lr']:
            scheduler.step()
            train_dataloader = bs_scheduler.step()

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, help='The name of the experiment', default='experiment_disentangled_cotgen')
    parser.add_argument('--config', type=str, help='The path to the config file', default='./config_disentangled_sq.yaml')
    parser.add_argument('--device', type=str, help='cpu or cuda', default='cuda')
    parser.add_argument('--load_from', type=str, help='continue training', default=None)
    parser.add_argument('--scenes_path', type=str, help='folder of scenes', default='data_cogent/')
    parser.add_argument('--questions_path', type=str, help='folder of questions', default='data_cogent/')
    parser.add_argument('--clvr_path', type=str, help='folder before images', default='data_cogent/')
    parser.add_argument('--use_cache', type=int, help='if to use cache (only in image clever)', default=0)
    parser.add_argument('--use_hdf5', type=int, help='if to use hdf5 loader', default=0)
    parser.add_argument('--freeze_exponential_growth', type=int, help='if to stay on same lr uppon resume', default=0)
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

    train_model(config=args.config, device=args.device, experiment_name=args.name, load_from=args.load_from,
                scenes_path=args.scenes_path, questions_path=args.questions_path, clvr_path=args.clvr_path,
                use_cache=args.use_cache, use_hdf5=args.use_hdf5,
                freeze_exponential_growth=args.freeze_exponential_growth)
