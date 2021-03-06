import os
import os.path as osp
import sys

import numpy.random
import torch.nn

from deltalogger.deltalogger import Deltalogger
from reinforce_modules.utils import ConfusionGame, get_defense_visual_fool_model
from utils.train_utils import StateCLEVR, ImageCLEVR_HDF5

sys.path.insert(0, osp.abspath('.'))
import random
import argparse
from modules.embedder import *
import seaborn as sns
from reinforce_modules.policy_networks import Re1nforceTrainer, PolicyNet

sns.set_style('darkgrid')


def _print(something):
    print(something, flush=True)
    return


def PolicyEvaluation(args, seed=1, logger=None):
    torch.manual_seed(seed)
    random.seed(seed)
    numpy.random.seed(seed)
    effective_range_name = 'all'
    BS = args.bs
    TRAIN_DURATION = args.train_duration

    if osp.exists(f'./results/experiment_reinforce_increasing_defense_{args.defense_level}'):
        pass
    else:
        os.mkdir(f'./results/experiment_reinforce_increasing_defense_{args.defense_level}')

    if args.backend == 'states':
        load_from = './results/experiment_rn/mos_epoch_164.pt'
    else:
        load_from = './results/experiment_fp/mos_epoch_219.pt'

    model, (
        model_fool,
        resnet), val_dataloader, predictions_before_pre_calc, initial_example = get_defense_visual_fool_model(
        device=args.device,
        load_from=load_from,
        scenes_path=args.scenes_path,
        questions_path=args.questions_path,
        clvr_path=args.clvr_path,
        batch_size=BS,
        defense_level=args.defense_level)

    rl_game = ConfusionGame(testbed_model=model,
                            confusion_model=model_fool,
                            device='cuda',
                            batch_size=BS,
                            confusion_weight=args.confusion_weight,
                            change_weight=args.change_weight,
                            fail_weight=args.fail_weight,
                            invalid_weight=args.invalid_weight,
                            mode=args.mode,
                            render=args.mode == 'visual',
                            backend=args.backend)

    if args.backend == 'states':
        input_size = 512
    elif args.backend == 'pixels':
        input_size = 256
    else:
        raise ValueError(f"Backend must be [states/pixels] you entered: {args.backend}")
    model = PolicyNet(input_size=input_size, hidden_size=512, dropout=0.0, reverse_input=True)

    trainer = Re1nforceTrainer(model=model,
                               game=rl_game,
                               dataloader=val_dataloader,
                               device=args.device,
                               lr=args.lr,
                               train_duration=TRAIN_DURATION,
                               batch_size=BS,
                               name=effective_range_name,
                               predictions_before_pre_calc=predictions_before_pre_calc,
                               resnet=resnet,
                               fool_model_name='Defense',
                               initial_example=initial_example)

    best_drop, best_confusion = trainer.train(log_every=-1, save_every=100, logger=logger)
    return best_drop, best_confusion


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, help='cpu or cuda', default='cuda')
    parser.add_argument('--scenes_path', type=str, help='folder of scenes', default='data/')
    parser.add_argument('--questions_path', type=str, help='folder of questions', default='data/')
    parser.add_argument('--clvr_path', type=str, help='folder before images', default='data/')
    parser.add_argument('--use_cache', type=int, help='if to use cache (only in image clever)', default=0)
    parser.add_argument('--use_hdf5', type=int, help='if to use hdf5 loader', default=0)
    parser.add_argument('--confusion_weight', type=float, help='what kind of experiment to run', default=1)
    parser.add_argument('--change_weight', type=float, help='what kind of experiment to run', default=0.1)
    parser.add_argument('--fail_weight', type=float, help='what kind of experiment to run', default=-0.1)
    parser.add_argument('--invalid_weight', type=float, help='what kind of experiment to run', default=-0.8)
    parser.add_argument('--train_duration', type=int, help='what kind of experiment to run', default=30)
    parser.add_argument('--lr', type=float, help='what kind of experiment to run', default=5e-4)
    parser.add_argument('--bs', type=int, help='what kind of experiment to run', default=10)
    parser.add_argument('--mode', type=str, help='state | visual | imagenet', default='visual')
    parser.add_argument('--range', type=float, default=-1)
    parser.add_argument('--seed', type=int, default=51)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--backend', type=str, help='states or pixels', default='states')
    parser.add_argument('--defense_level', type=int, default=10)

    args = parser.parse_args()


    if args.backend == 'states':
        exp_name = 'DeltaDefense'
    elif args.backend == 'pixels':
        exp_name = 'DeltaDefensePixels'
    else:
        raise ValueError(f'Backend has to be one of states/pixels, you entered : {args.backend}')
    if args.repeat == 1:

        logger = Deltalogger(exp_name, run_tag=[args.defense_level, 0], dummy=True)

        _print(PolicyEvaluation(args, args.seed, logger=logger))
    else:
        acc_drops = []
        cons_drops = []
        for seed in range(args.seed, args.repeat + args.seed):
            experiment_number = seed - args.seed
            logger = Deltalogger(exp_name, run_tag=[args.defense_level, experiment_number],
                                 dummy=False)

            a, c = PolicyEvaluation(args, seed, logger=logger)
            acc_drops.append(a)
            cons_drops.append(c)
        _print(f'Accuracy: Min: {min(acc_drops)}, Mean: {sum(acc_drops) / len(acc_drops)}, Max: {max(acc_drops)}')
        _print(
            f'Consistency: Min: {min(cons_drops)}, Mean: {sum(cons_drops) / len(cons_drops)}, Max: {max(cons_drops)}')
