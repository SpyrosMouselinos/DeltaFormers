import os
import os.path as osp
import sys
from shutil import copyfile

import yaml

sys.path.insert(0, osp.abspath('.'))

import argparse
from torch.utils.data import Dataset
from modules.embedder import *
from utils.train_utils import StateCLEVR, ImageCLEVR, BatchSizeScheduler

AVAILABLE_DATASETS = {
    'DeltaRN': StateCLEVR,
    'DeltaSQFormer': StateCLEVR,
    'DeltaQFormer': StateCLEVR,
    'DeltaRNFP': ImageCLEVR,
}

AVAILABLE_MODELS = {'DeltaRN': DeltaRN,
                    'DeltaRNFP': DeltaRNFP,
                    'DeltaSQFormer': DeltaSQFormer,
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
    print("Remember that available modes are: [all, model, model+opt, model+opt+sched]\n")
    checkpoint = torch.load(path)
    epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model_state_dict'])
    if ('opt' in mode or 'all' in mode) and optim is not None:
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        print("Optimizer not Loaded!\n")

    if ('sched' in mode or 'all' in mode) and optim is not None:
        sched.load_state_dict(checkpoint['scheduler_state_dict'])
    else:
        print("Scheduler not Loaded!\n")

    if ('bs_sched' in mode or 'all' in mode) and optim is not None:
        bs_sched.load_state_dict(checkpoint['bs_scheduler_state_dict'])
    else:
        print("BS Scheduler not Loaded!\n")
    print(f"Your model achieves {round(checkpoint['val_loss'], 4)} validation loss\n")
    return model, optim, sched, bs_sched, epoch


def kwarg_dict_to_device(data_obj, device):
    if device == 'cpu':
        return data_obj
    cpy = {}
    for key, _ in data_obj.items():
        cpy[key] = data_obj[key].to(device)
    return cpy


def check_paths(experiment_name):
    if osp.exists('results'):
        pass
    else:
        os.mkdir(f'results/')

    if osp.exists(f'results/{experiment_name}'):
        pass
    else:
        os.mkdir(f'results/{experiment_name}')

    copyfile('config_sq.yaml', f'./results/{experiment_name}/config.yaml')
    return


def accuracy_metric(y_pred, y_true):
    acc = (y_pred.argmax(1) == y_true).float().detach().cpu().numpy()
    return float(100 * acc.sum() / len(acc))


def train_model(config, device, experiment_name='experiment_1', load_from=None):
    if device == 'cuda':
        device = 'cuda:0'

    if load_from is not None:
        experiment_name = load_from.split('results/')[-1].split('/')[0]
        config = f'./results/{experiment_name}/config.yaml'
        with open(config, 'r') as fin:
            config = yaml.load(fin, Loader=yaml.FullLoader)

        model = AVAILABLE_MODELS[config['model_architecture']](config)
        print(f"Loading Model of type: {config['model_architecture']}\n", flush=True)
        model = model.to(device)
        model.train()
        #TODO: Change this!
        train_set = AVAILABLE_DATASETS[config['model_architecture']](config=config, split='train')

        train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=config['batch_size'], shuffle=True)

        val_set = AVAILABLE_DATASETS[config['model_architecture']](config=config, split='val')

        val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=config['batch_size'], shuffle=False)

        optimizer = torch.optim.Adam(params=model.parameters(), lr=config['lr'], weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['scheduler_step_size'],
                                                    gamma=config['scheduler_gamma'])

        bs_scheduler = BatchSizeScheduler(train_ds=train_set, initial_bs=config['batch_size'],
                                          step_size=config['scheduler_step_size'], gamma=config['scheduler_gamma'],
                                          max_bs=config['max_batch_size'])
        model, optimizer, scheduler, bs_scheduler, init_epoch = load(path=load_from, model=model, optim=optimizer,
                                                                     sched=scheduler, bs_sched=bs_scheduler,
                                                                     mode='all')

    else:
        check_paths(experiment_name)
        with open(config, 'r') as fin:
            config = yaml.load(fin, Loader=yaml.FullLoader)
        model = AVAILABLE_MODELS[config['model_architecture']](config)
        print(f"Loading Model of type: {config['model_architecture']}\n", flush=True)
        model = model.to(device)
        model.train()
        # TODO: Change this!

        train_set = AVAILABLE_DATASETS[config['model_architecture']](config=config, split='train')
        train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=config['batch_size'], shuffle=True)

        val_set = AVAILABLE_DATASETS[config['model_architecture']](config=config, split='val')
        val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=config['batch_size'], shuffle=False)

        optimizer = torch.optim.Adam(params=model.parameters(), lr=config['lr'], weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['scheduler_step_size'],
                                                    gamma=config['scheduler_gamma'])

        bs_scheduler = BatchSizeScheduler(train_ds=train_set, initial_bs=config['batch_size'],
                                          step_size=config['bs_scheduler_step_size'],
                                          gamma=config['bs_scheduler_gamma'],
                                          max_bs=config['max_batch_size'])
        init_epoch = 0

    criterion = nn.CrossEntropyLoss()
    metric = accuracy_metric

    total_loss = 0.
    total_acc = 0.
    best_val_loss = 1000
    overfit_count = -3
    log_interval = config['log_every']
    for epoch in range(init_epoch, config['max_epochs']):
        for train_batch_index, train_batch in enumerate(train_dataloader):
            if ((epoch + 1) * train_batch_index) % config['validate_every'] == 0 and train_batch_index > 0:
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
                print('| epoch {:3d} | val loss {:5.2f} | val acc {:5.2f} \n'.format(epoch,
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
                        print(f"Training stopped at epoch: {epoch} and best validation loss: {best_val_loss}")
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
                    print('| epoch {:3d} | {:5d}/{:5d} batches | '
                          'lr {:02.6f} | loss {:5.2f} | acc {:5.2f}\n'.format(epoch, train_batch_index,
                                                                              len(train_dataloader),
                                                                              scheduler.get_last_lr()[0], cur_loss,
                                                                              cur_acc))
            # End of batch #
        # End of epoch #
        total_loss = 0.
        total_acc = 0.
        if scheduler.get_last_lr()[0] < config['max_lr']:
            scheduler.step()
            train_dataloader = bs_scheduler.step()

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, help='The name of the experiment', default='experiment_rn_fp')
    parser.add_argument('--config', type=str, help='The path to the config file', default='./config_rn_fp.yaml')
    parser.add_argument('--device', type=str, help='cpu or cuda', default='cuda')
    parser.add_argument('--load_from', type=str, help='continue training', default=None)
    args = parser.parse_args()
    train_model(config=args.config, device=args.device, experiment_name=args.name, load_from=args.load_from)
