import sys
sys.path.insert(0, './bert_modules/')

import yaml
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module as Module
import matplotlib.pyplot as plt
from bert_modules import BertEncoder, BertLayerNorm

class MultiModalEmbedder(Module):
    def __init__(self, config: dict):
        super(MultiModalEmbedder,self).__init__()
        self.question_embeddings    = nn.Embedding(config['question_vocabulary_size'], config['hidden_dim'], padding_idx=0)
        self.answer_embeddings      = nn.Embedding(1, config['hidden_dim'])
        self.position_embeddings    = nn.Embedding(config['max_objects_per_scene'] + config['question_vocabulary_size'] + 1, config['hidden_dim'])
        self.token_type_embeddings  = nn.Embedding(config['n_token_types'], config['hidden_dim'], padding_idx=0)
        self.color_embeddings       = nn.Embedding(4, config['embedding_dim'])
        self.shape_embeddings       = nn.Embedding(4, config['embedding_dim'])
        self.material_embeddings    = nn.Embedding(4, config['embedding_dim'])
        self.size_embeddings        = nn.Embedding(4, config['embedding_dim'])
        
        self.position_project = nn.Linear(3, config['embedding_dim'])
        self.scene_project    = nn.Linear(3, config['hidden_dim'])
        self.reproject        = nn.Linear(5 * config['embedding_dim'], config['hidden_dim'])
        self.LayerNorm        = BertLayerNorm(config['hidden_dim'], eps=1e-12)
        self.dropout          = nn.Dropout(0.1)
        return
    
    def forward(self,
                active_positions=None,
                active_modes=None,
                object_positions=None,
                object_colors=None,
                object_shapes=None,
                object_materials=None,
                object_sizes=None,
                scene_state=None,
                questions=None,
                answers=None):

        ### Generate active positions ###
        positions = torch.cat([torch.LongTensor(list(range(0, 61))).view(1,61) for _ in range(0,16)], 0)
        position_embeddings = self.position_embeddings(positions)
        
        
        ### Generate types ###              #--4-- Items--- 6 empty --- 1 scene-- 50 word question -- #
        types = torch.cat([torch.LongTensor([1,1,1,1,0,0,0,0,0,0,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
                                             2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
                                             2,2,2,2,2,2,2,2,2,2,2,2]).view(1,61) for _ in range(0,16)], 0)
        type_embeddings = self.token_type_embeddings(types)
        
        
        ### Get Tokenized and Padded Questions ###
        ### BS X Q_SEQ_LEN 
        questions = torch.cat([torch.LongTensor(list(range(0, 50))).view(1,50) for _ in range(0,16)], 0)
        ### BS X Q_SEQ_LEN X Reproj Emb
        questions = self.question_embeddings(questions)
        
        # ### Get Tokenized and Padded Answers ###
        # ### BS X 1 
        # answers  = torch.cat([torch.LongTensor([5]).view(1,1) for _ in range(0,16)], 0)
        # ### BS X 1 X Reproj Emb
        # answers  = self.answer_embeddings(answers)
        
        ### Generate Attention Mask ###
        mask = types.ge(1) * 1.0
        mask = mask.unsqueeze(1).unsqueeze(2)
        mask = (1.0 - mask) * -10000.0
        
        ### Generate Object Mask ###
        object_mask = types.eq(1) * 1.0
        
        ### Gather all positions ###
        ### BS X SEQ_LEN X 3 DIM
        op = torch.rand(size=(16,10,3))
        
        ### Projected Version ###
        ### BS X SEQ_LEN X EMB_DIM
        op_proj = torch.tanh(self.position_project(op))
        
        ### Gather scene state ###
        ### BS X SEQ_LEN x 3 DIM
        ss = torch.rand(size=(16,1,3))

        ### Projected Version ###
        ### BS X SEQ_LEN X EMB_DIM
        ss_proj = torch.tanh(self.scene_project(ss))
        
        ### Gather all colors ###
        oc = torch.randint(high=4, size=(16,10), dtype=torch.long)
        
        ### Get embeddings ###
        oc_proj =  self.color_embeddings(oc)
        
        ### Gather all shapes ###
        os = torch.randint(high=4, size=(16,10), dtype=torch.long)
        
        ### Get embeddings ###
        os_proj =  self.shape_embeddings(os)
        
        ### Gather all materials ###
        om = torch.randint(high=4, size=(16,10), dtype=torch.long)

        ### Get embeddings ###
        om_proj =  self.material_embeddings(om)
        
        ### Gather all sizes ###
        oz = torch.randint(high=4, size=(16,10), dtype=torch.long)

        ### Get embeddings ###
        oz_proj =  self.size_embeddings(oz)

        object_related_embeddings = torch.cat([op_proj, oc_proj, os_proj, om_proj, oz_proj],2)
        
        ### Reproject them to 128 sized-embeddings ###
        ore = self.reproject(object_related_embeddings)
        
        ### Stack all embeddings on the sequence dimension ###
        pre_augmentation_embeddings = torch.cat([ore, ss_proj, questions], 1)
        augmentation_embeddings = position_embeddings + type_embeddings
        
        embeddings = pre_augmentation_embeddings + augmentation_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings, mask, object_mask
    
def test_embedder():
    with open(osp.dirname(osp.dirname(__file__)) + '/config.yaml', 'r') as fin:
        config =yaml.load(fin, Loader=yaml.FullLoader)
        
        
    correct = MultiModalEmbedder(config)
    e,m,om = correct.forward()
    test_bert = BertEncoder(config)
    out = test_bert.forward(e, m, output_all_encoded_layers=False, output_attention_probs=False)
    out = out[0]
    ### Get the Item Outputs - Token 0-10 ##
    print(out[:,0:10].size())
    
    ### Filter By Object Mask ###
    print(out * om.unsqueeze(2))
    
    ### Get the Scene Output - Token 11 - ##
    print(out[:,10].size())



    
test_embedder()