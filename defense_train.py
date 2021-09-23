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
    adversarial_agent_load_from_1 = args.aalf1
    adversarial_agent_load_from_2 = args.aalf2
    effective_range_1 = args.range_1 * 1000
    effective_range_2 = args.range_2 * 1000
    if effective_range_2 is None:
        effective_range = effective_range_1
    else:
        effective_range = [effective_range_1, effective_range_2]
    effective_range_1_offset = args.range_1_offset
    effective_range_2_offset = args.range_2_offset

    if effective_range_2_offset is None:
        effective_range_offset = effective_range_1_offset
    else:
        effective_range_offset = [effective_range_1_offset, effective_range_2_offset]

    if adversarial_agent_load_from_2 is None:
        adversarial_agent_load_from = adversarial_agent_load_from_1
    else:
        adversarial_agent_load_from = [adversarial_agent_load_from_1, adversarial_agent_load_from_2]

    feature_extractor_load_from = args.felf
    vqa_model_load_type = args.vmlt

    adversarial_agent, feature_extractor, vqa_agent, resnet, minigames = get_defense_models(
        adversarial_agent_load_from=adversarial_agent_load_from,
        feature_extractor_load_from=feature_extractor_load_from,
        vqa_model_load_type=vqa_model_load_type,
        batch_size=bs,
        effective_range=effective_range,
        effective_range_offset=effective_range_offset,
        randomize_range=False)

    train_range = f'{args.range_1_offset}_{args.range_1_offset + args.range_1 * 1000}'
    train_name = args.vmlt
    defense_game = DefenseGame(vqa_model=vqa_agent,
                               adversarial_agent_1=adversarial_agent[0],
                               adversarial_agent_2=adversarial_agent[1],
                               feature_extractor_backbone=feature_extractor,
                               resnet=resnet,
                               device='cuda',
                               batch_size=5,
                               defense_rounds=defense_rounds,
                               pipeline='extrapolation',
                               mode='visual', train_name=train_name, train_range=train_range, use_mixed_data=args.use_mix == 'True')
    defense_game.engage(minigame=minigames[0], minigame2=minigames[1], vqa_model=None, adversarial_agent=None,
                        adversarial_agent_eval=None, train_vqa=True, train_agent=False, but_list=['fc1'])

    defense_game.assess_overall_drop(train_name, train_range)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, help='Batch Size', default=5)
    parser.add_argument('--defense_rounds', type=int, help='Rounds / Epochs of Defense', default=1000)
    parser.add_argument('--aalf1', type=str, help='Adversarial Agent Load From 1',
                        default='./results/experiment_reinforce/visual/model_reinforce_0.1k_rnfp_0.pt')
    parser.add_argument('--aalf2', type=str, help='Adversarial Agent Load From 2',
                        default='./results/experiment_reinforce/visual/model_reinforce_0.1k_rnfp_100.pt')
    parser.add_argument('--felf', type=str, help='Feature Extractor Load From',
                        default='./results/experiment_rn/mos_epoch_164.pt')
    parser.add_argument('--vmlt', type=str, help='VQA Load Type', default='rnfp')

    parser.add_argument('--range_1_offset', type=float, default=0)
    parser.add_argument('--range_2_offset', type=float, default=100)
    parser.add_argument('--range_1', type=float, default=0.1)
    parser.add_argument('--range_2', type=float, default=0.1)
    parser.add_argument('--randomize_range', type=str, default='False')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--use_mix', type=str, default='False')

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
