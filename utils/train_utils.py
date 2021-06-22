import torch
import torch.nn as nn
import numpy as np

def mock_input():
    positions = torch.cat([torch.LongTensor(list(range(0, 62))).view(1,62) for _ in range(0,16)], 0)
    types = torch.cat([torch.LongTensor([2,1,1,1,1,0,0,0,0,0,0,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
                                            2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
                                            2,2,2,2,2,2,2,2,2,2,2,2]).view(1,62) for _ in range(0,16)], 0)
    questions = torch.cat([torch.LongTensor(list(range(0, 50))).view(1,50) for _ in range(0,16)], 0)
    object_positions = torch.rand(size=(16,10,3))
    scene_state = torch.rand(size=(16,1,3))
    object_colors = torch.randint(high=4, size=(16,10), dtype=torch.long)
    object_shapes = torch.randint(high=4, size=(16,10), dtype=torch.long)
    object_materials = torch.randint(high=4, size=(16,10), dtype=torch.long)
    object_sizes = torch.randint(high=4, size=(16,10), dtype=torch.long)