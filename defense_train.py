import numpy.random
import torch.nn
import os
import os.path as osp
import sys
from deltalogger.deltalogger import Deltalogger
from reinforce_modules.utils import DefenseGame, get_defense_models
from utils.train_utils import MixCLEVR_HDF5

sys.path.insert(0, osp.abspath('.'))
import random
import argparse
from modules.embedder import *
import seaborn as sns


sns.set_style('darkgrid')


def _print(something):
    print(something, flush=True)
    return


def DefenseEvaluation(args, seed=1, logger=None):
    torch.manual_seed(seed)
    random.seed(seed)
    numpy.random.seed(seed)


    if args.range == -1:
        effective_range = None
        effective_range_name = 'all'
    else:
        effective_range = args.range * 1000
        effective_range_name = f'{args.range}k'
    if osp.exists(f'./results/experiment_defense'):
        pass
    else:
        os.mkdir(f'./results/experiment_defense')

    if osp.exists(f'./results/experiment_defense/visual'):
        pass
    else:
        os.mkdir(f'./results/experiment_defense/visual')

    bs = args.bs
    defense_rounds = args.defense_rounds
    adversarial_agent_load_from = args.aalf
    feature_extractor_load_from = args.felf
    vqa_model_load_type = args.vmlt

    adversarial_agent, feature_extractor, vqa_agent, resnet, minigame = get_defense_models(
        adversarial_agent_load_from=adversarial_agent_load_from,
        feature_extractor_load_from=feature_extractor_load_from,
        vqa_model_load_type=vqa_model_load_type,
        batch_size=bs,
        effective_range=effective_range,
        randomize_range=eval(args.randomize_range))

    defense_game = DefenseGame(vqa_model=vqa_agent,
                               adversarial_agent=adversarial_agent,
                               feature_extractor_backbone=feature_extractor,
                               resnet=resnet,
                               device='cuda',
                               batch_size=5,
                               defense_rounds=defense_rounds,
                               pipeline='extrapolation',
                               mode='visual')
    defense_game.clone_adversarial_agent(random_weights=False)
    defense_game.engage(minigame=minigame, vqa_model=None, adversarial_agent=None, train_vqa=True, train_agent=False)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, help='Batch Size', default=5)
    parser.add_argument('--defense_rounds', type=int, help='Rounds / Epochs of Defense', default=3)
    parser.add_argument('--aalf', type=str, help='Adversarial Agent Load From',
                        default='./results/experiment_reinforce/visual/model_reinforce_0.01k_rnfp.pt')
    parser.add_argument('--felf', type=str, help='Feature Extractor Load From',
                        default='./results/experiment_rn/mos_epoch_164.pt')
    parser.add_argument('--vmlt', type=str, help='VQA Load Type', default='rnfp')

    parser.add_argument('--range', type=float, default=0.01)
    parser.add_argument('--randomize_range', type=str, default='False')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--repeat', type=int, default=1)

    args = parser.parse_args()
    if args.repeat == 1:
        # TODO: DELETE THIS
        # logger = Deltalogger('DeltaFormers', run_tag=[args.fool_model, 1000 * args.range, 1], dummy=True)

        _print(DefenseEvaluation(args, args.seed, logger=None))
    else:
        acc_drops = []
        cons_drops = []
        for seed in range(args.seed, args.repeat + args.seed):
            experiment_number = seed - args.seed
            # TODO: DELETE THIS
            logger = Deltalogger('DeltaFormers', run_tag=[args.fool_model, 1000 * args.range, experiment_number],
                                 dummy=True)

            a, c = DefenseEvaluation(args, seed, logger=logger)
            acc_drops.append(a)
            cons_drops.append(c)
        _print(f'Final Results on {args.fool_model} for games of length: {args.range * 1000} for {args.repeat} RUNS:')
        _print(f'Accuracy: Min: {min(acc_drops)}, Mean: {sum(acc_drops) / len(acc_drops)}, Max: {max(acc_drops)}')
        _print(
            f'Consistency: Min: {min(cons_drops)}, Mean: {sum(cons_drops) / len(cons_drops)}, Max: {max(cons_drops)}')
