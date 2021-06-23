import time
import math
import sys
import os
import os.path as osp
from shutil import copyfile
sys.path.insert(0, osp.abspath('.'))

import argparse
import yaml
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from modules.embedder import DeltaFormer
from utils.train_utils import StateCLEVR


    
def check_paths(experiment_name):
    if osp.exists('./results'):
        pass
    
    if osp.exists(f'./results/{experiment_name}'):
        pass
    else:
        os.mkdir(f'./results/{experiment_name}')
        
    copyfile('./config.yaml', f'./results/{experiment_name}/config.yaml')
    return

    
def train_model(config, device, experiment_name='experiment_1'):
    check_paths(experiment_name)
    if device == 'cuda':
        device = 'cuda:0'
    train_dataloader = torch.utils.data.DataLoader(StateCLEVR(config=config, split='train'), batch_size=config['batch_size'], shuffle=True)
    val_dataloader   = torch.utils.data.DataLoader(StateCLEVR(config=config, split='val'), batch_size=config['batch_size'], shuffle=False)
    
    model = DeltaFormer(config)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr = config['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['scheduler_step_size'], gamma=config['scheduler_gamma'])
    
    total_loss = 0.
    log_interval = config['log_every']
    for epoch in range(0, config['max_epochs']):
        for train_batch_index, train_batch in enumerate(train_dataloader):
            
            ### If batch index is equal to validation limit, validate ###
            if train_batch_index % config['validate_every'] == 0 and train_batch_index > 0:
                total_val_loss = 0.
                # Turn off the train mode #
                model.eval()
                with torch.no_grad():
                    for val_batch_index, val_batch in enumerate(val_dataloader):
                        data, y_real = val_batch
                        y_pred = model(data)
                        val_loss = criterion(y_pred, y_real)
                        total_val_loss += val_loss.item()
                print('| epoch {:3d} | val loss {:5.2f}\n').format(epoch, total_val_loss / val_batch_index)
            else:
                # Turn on the train mode #
                model.train() 
                data, y_real = train_batch
                optimizer.zero_grad()
                
                y_pred = model(data)
                loss = criterion(y_pred, y_real)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                
                if train_batch_index % log_interval == 0 and train_batch_index > 0:
                    cur_loss = total_loss / log_interval
                    print('| epoch {:3d} | {:5d}/{:5d} batches | '
                        'lr {:02.2f} | loss {:5.2f}\n'.format(epoch, train_batch_index, len(train_dataloader), scheduler.get_lr()[0], cur_loss))
                    total_loss = 0
            # End of batch #
        # End of epoch #
        scheduler.step()
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, help='The name of the experiment', default='experiment_1')
    parser.add_argument('--config', type=str, help='The path to the config file', default='config.yaml')
    parser.add_argument('--device', type=str, help='cpu or cuda', default='cpu')
    args = parser.parse_args()
    train_model(config=args.config, device=args.device, experiment_name=args.name)
    
    





def evaluate(eval_model, data_source):
    with torch.no_grad():
        for i in range(0, nbatches, bptt):
            data, targets = get_batch(data_source, i)
            output = eval_model(data).local_value()
            output_flat = output.view(-1, ntokens)
            # Need to move targets to the device where the output of the
            # pipeline resides.
            total_loss += len(data) * criterion(output_flat, targets.cuda(1)).item()
    return total_loss / (len(data_source) - 1)