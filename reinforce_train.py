import numpy.random
import torch.nn
import os
import os.path as osp
import sys
from deltalogger.deltalogger import Deltalogger
from reinforce_modules.utils import get_fool_model, get_visual_fool_model, ConfusionGame

sys.path.insert(0, osp.abspath('.'))
import random
import argparse
from modules.embedder import *
import seaborn as sns
from reinforce_modules.policy_networks import Re1nforceTrainer, ImageNetPolicyNet, PolicyNet

sns.set_style('darkgrid')


def _print(something):
    print(something, flush=True)
    return


def PolicyEvaluation(args, seed=1, logger=None):
    torch.manual_seed(seed)
    random.seed(seed)
    numpy.random.seed(seed)
    if args.mode == 'state':
        prefix = 'state'
    else:
        prefix = 'visual'

    if args.range == -1:
        effective_range = None
        effective_range_name = 'all'
    else:
        effective_range = args.range * 1000
        effective_range_name = f'{args.range}k'
    if osp.exists(f'./results/experiment_reinforce/{prefix}'):
        pass
    else:
        os.mkdir(f'./results/experiment_reinforce/{prefix}')
    BS = args.bs

    if args.backend == 'states':
        load_from = './results/experiment_rn/mos_epoch_164.pt'
    else:
        load_from = './results/experiment_fp/mos_epoch_219.pt'

    if prefix == 'state':
        model, model_fool, val_dataloader = get_fool_model(device=args.device, load_from=load_from,
                                                           scenes_path=args.scenes_path,
                                                           questions_path=args.questions_path,
                                                           clvr_path=args.clvr_path,
                                                           use_cache=args.use_cache,
                                                           use_hdf5=args.use_hdf5,
                                                           batch_size=BS,
                                                           mode=args.mode,
                                                           effective_range=effective_range,
                                                           mos_epoch=args.mos_epoch,
                                                           randomize_range=eval(args.randomize_range))
        predictions_before_pre_calc = None
        resnet = None
        initial_example = None

    else:
        if args.range_offset is not None:
            range_offset = int(args.range_offset)
        else:
            range_offset = None
        model, (model_fool, resnet), val_dataloader, predictions_before_pre_calc, initial_example = get_visual_fool_model(
            device=args.device,
            load_from=load_from,
            scenes_path=args.scenes_path,
            questions_path=args.questions_path,
            clvr_path=args.clvr_path,
            use_cache=args.use_cache,
            use_hdf5=args.use_hdf5,
            batch_size=BS,
            mode=args.mode,
            effective_range=effective_range,
            fool_model=args.fool_model, randomize_range=eval(args.randomize_range), range_offset=range_offset)

    train_duration = args.train_duration
    rl_game = ConfusionGame(testbed_model=model, confusion_model=model_fool, device='cuda', batch_size=BS,
                            confusion_weight=args.confusion_weight, change_weight=args.change_weight,
                            fail_weight=args.fail_weight, invalid_weight=args.invalid_weight, mode=args.mode,
                            render=args.mode == 'visual', backend=args.backend)
    if args.mode == 'state' or args.mode == 'visual':
        if args.backend == 'states':
            input_size = 512
        elif args.backend == 'pixels':
            input_size = 256
        else:
            raise ValueError(f"Backend must be [states/pixels] you entered: {args.backend}")
        model = PolicyNet(input_size=input_size, hidden_size=512, dropout=0.0, reverse_input=True)
    elif args.mode == 'imagenet':
        model = ImageNetPolicyNet(input_size=128, hidden_size=256, dropout=0.0, reverse_input=True)
    else:
        raise ValueError
    if args.cont > 0:
        print("Loading model...")
        model.load(f'./results/experiment_reinforce/visual/model_reinforce_0.01k_rnfp.pt')

    if args.mode == 'visual':
        fool_model_name = args.fool_model
    else:
        fool_model_name = 'LinAttFormer'
    trainer = Re1nforceTrainer(model=model, game=rl_game, dataloader=val_dataloader, device=args.device, lr=args.lr,
                               train_duration=train_duration, batch_size=BS, name=effective_range_name,
                               predictions_before_pre_calc=predictions_before_pre_calc, resnet=resnet,
                               fool_model_name=fool_model_name, initial_example=initial_example)

    best_drop, best_confusion = trainer.train(log_every=-1, save_every=100, logger=logger)
    #trainer.evaluate(example_range=(0, 100))
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
    parser.add_argument('--train_duration', type=int, help='what kind of experiment to run', default=100)
    parser.add_argument('--lr', type=float, help='what kind of experiment to run', default=5e-3)
    parser.add_argument('--bs', type=int, help='what kind of experiment to run', default=5)
    parser.add_argument('--cont', type=int, help='what kind of experiment to run', default=0)
    parser.add_argument('--mode', type=str, help='state | visual | imagenet', default='visual')
    parser.add_argument('--range', type=float, default=0.01)
    # TODO: DELETE THIS
    # TODO: DELETE THIS
    parser.add_argument('--randomize_range', type=str, default='False')
    parser.add_argument('--range_offset', type=int, default=0)
    # TODO: DELETE THIS
    # TODO: DELETE THIS
    parser.add_argument('--mos_epoch', type=int, default=164)
    parser.add_argument('--fool_model', type=str, default='rnfp')
    parser.add_argument('--seed', type=int, default=543)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--backend', type=str, help='states or pixels', default='states')

    args = parser.parse_args()

    if args.backend == 'states':
        exp_name = 'DeltaFormers'
    elif args.backend == 'pixels':
        exp_name = 'DeltaFormersPixels'
    else:
        raise ValueError(f'Backend has to be one of states/pixels, you entered : {args.backend}')
    if args.repeat == 1:
        _print(f'Final Results on {args.fool_model}:')

        logger = Deltalogger(exp_name, run_tag=[args.fool_model, 1000 * args.range, 1], dummy=True)

        _print(PolicyEvaluation(args, args.seed, logger=logger))
    else:
        acc_drops = []
        cons_drops = []
        for seed in range(args.seed, args.repeat + args.seed):
            experiment_number = seed - args.seed
            logger = Deltalogger(exp_name, run_tag=[args.fool_model, 1000 * args.range, experiment_number],
                                 dummy=False)

            a, c = PolicyEvaluation(args, seed, logger=logger)
            acc_drops.append(a)
            cons_drops.append(c)
        _print(f'Final Results on {args.fool_model} for games of length: {args.range * 1000} for {args.repeat} RUNS:')
        _print(f'Accuracy: Min: {min(acc_drops)}, Mean: {sum(acc_drops) / len(acc_drops)}, Max: {max(acc_drops)}')
        _print(
            f'Consistency: Min: {min(cons_drops)}, Mean: {sum(cons_drops) / len(cons_drops)}, Max: {max(cons_drops)}')
