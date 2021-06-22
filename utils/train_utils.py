import sys
sys.path.insert(0, './modules/')

import yaml
import os.path as osp
import torch
import torch.nn as nn
from embedder import DeltaFormer

def mock_input(batch_size):
    positions = torch.cat([torch.LongTensor(list(range(0, 62))).view(1,62) for _ in range(0,batch_size)], 0)
    types = torch.cat([torch.LongTensor([2,1,1,1,1,0,0,0,0,0,0,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
                                            2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
                                            2,2,2,2,2,2,2,2,2,2,2,2]).view(1,62) for _ in range(0,batch_size)], 0)
    questions = torch.cat([torch.LongTensor(list(range(0, 50))).view(1,50) for _ in range(0,batch_size)], 0)
    object_positions = torch.rand(size=(batch_size,10,3))
    scene_state = torch.rand(size=(batch_size,1,3))
    object_colors = torch.randint(high=4, size=(batch_size,10), dtype=torch.long)
    object_shapes = torch.randint(high=4, size=(batch_size,10), dtype=torch.long)
    object_materials = torch.randint(high=4, size=(batch_size,10), dtype=torch.long)
    object_sizes = torch.randint(high=4, size=(batch_size,10), dtype=torch.long)
    with open(osp.dirname(osp.dirname(__file__)) + '/config.yaml', 'r') as fin:
        config =yaml.load(fin, Loader=yaml.FullLoader)
    model = Deltaformer(config)
    model()