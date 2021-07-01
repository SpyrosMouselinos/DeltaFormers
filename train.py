import os
import os.path as osp
import sys
from shutil import copyfile

import yaml

sys.path.insert(0, osp.abspath('.'))

import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from modules.embedder import DeltaSQFormer
from utils.train_utils import StateCLEVR


def save_all(model, optim, sched, epoch, loss,  path):
    torch.save({
        'epoch': epoch,
        'val_loss': loss,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        'scheduler_state_dict': sched.state_dict()
    }, path + f'/mos_epoch_{epoch}.pt')
    return


def load(path: str, model: nn.Module, optim=None, sched=None, mode='all'):
    print("Remember that available modes are: [all, model, model+opt, model+opt+sched]\n")
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if ('opt' in mode or 'all' in mode) and optim is not None:
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        print("Optimizer not Loaded!\n")

    if ('sched' in mode or 'all' in mode) and optim is not None:
        sched.load_state_dict(checkpoint['scheduler_state_dict'])
    else:
        print("Scheduler not Loaded!\n")
    print(f"Your model achieves {checkpoint['val_loss']} validation accuracy\n")
    return model, optim, sched


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

    copyfile('./config.yaml', f'./results/{experiment_name}/config.yaml')
    return


def accuracy_metric(y_pred, y_true):
    acc = (y_pred.argmax(1) == y_true).float().detach().cpu().numpy()
    return float(100 * acc.sum() / len(acc))


def train_model(config, device, experiment_name='experiment_1'):
    check_paths(experiment_name)
    if device == 'cuda':
        device = 'cuda:0'
    with open(config, 'r') as fin:
        config = yaml.load(fin, Loader=yaml.FullLoader)
    train_dataloader = torch.utils.data.DataLoader(StateCLEVR(config=config, split='train'),
                                                   batch_size=config['batch_size'], shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(StateCLEVR(config=config, split='val'),
                                                batch_size=config['batch_size'], shuffle=False)

    model = DeltaSQFormer(config)
    model = model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    metric = accuracy_metric
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['scheduler_step_size'],
                                                gamma=config['scheduler_gamma'])

    total_loss = 0.
    total_acc = 0.
    best_val_loss = 1000
    overfit_count = -1
    log_interval = config['log_every']
    for epoch in range(0, config['max_epochs']):
        for train_batch_index, train_batch in enumerate(train_dataloader):
            if ((epoch + 1) * train_batch_index) % config['validate_every'] == 0 and train_batch_index > 0:
                pass
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
                        break
                print('| epoch {:3d} | val loss {:5.2f} | val acc {:5.2f} \n'.format(epoch,
                                                                                     total_val_loss / (
                                                                                             val_batch_index + 1),
                                                                                     total_val_acc / (
                                                                                             val_batch_index + 1)))
                if total_val_loss / (val_batch_index + 1) < best_val_loss:
                    best_val_loss = total_val_loss / (val_batch_index + 1)
                    save_all(model, optimizer, scheduler, epoch, best_val_loss, f'./results/{experiment_name}')
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
                optimizer.step()

                total_loss += loss.item()
                total_acc += acc
                if train_batch_index % log_interval == 0 and train_batch_index > 0:
                    cur_loss = total_loss / (train_batch_index + 1)
                    cur_acc = total_acc / (train_batch_index + 1)
                    print('| epoch {:3d} | {:5d}/{:5d} batches | '
                          'lr {:02.4f} | loss {:5.2f} | acc {:5.2f}\n'.format(epoch, train_batch_index,
                                                                              len(train_dataloader),
                                                                              scheduler.get_last_lr()[0], cur_loss,
                                                                              cur_acc))
            # End of batch #
        # End of epoch #
        total_loss = 0.
        total_acc = 0.
        scheduler.step()

        if epoch % config['save_every_epochs'] == 0 and epoch > 0:
            save_all(model, optimizer, scheduler, epoch, best_val_loss, f'./results/{experiment_name}')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, help='The name of the experiment', default='experiment_sq_avg')
    parser.add_argument('--config', type=str, help='The path to the config file', default='./config.yaml')
    parser.add_argument('--device', type=str, help='cpu or cuda', default='cuda')
    args = parser.parse_args()
    train_model(config=args.config, device=args.device, experiment_name=args.name)
