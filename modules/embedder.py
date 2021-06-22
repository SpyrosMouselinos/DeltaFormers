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
        self.config = config
        self.question_embeddings    = nn.Embedding(config['question_vocabulary_size'], config['hidden_dim'], padding_idx=0)
        #self.answer_embeddings      = nn.Embedding(1, config['hidden_dim'])
        self.position_embeddings    = nn.Embedding(1 + config['max_objects_per_scene'] + config['question_vocabulary_size'] + 1, config['hidden_dim'])
        self.token_type_embeddings  = nn.Embedding(config['num_token_types'], config['hidden_dim'], padding_idx=0)
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
                positions,
                types,
                object_positions,
                object_colors,
                object_shapes,
                object_materials,
                object_sizes,
                scene_state,
                questions,
                answers=None):

        ### Generate active positions ###
        position_embeddings = self.position_embeddings(positions)
        
        
        ### Generate types ###              #1 CLS -- --4-- Items--- 6 empty --- 1 scene-- 50 word question -- #
        type_embeddings = self.token_type_embeddings(types)
        
        
        ### Get Tokenized and Padded Questions ###
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
        ### Projected Version ###
        ### BS X SEQ_LEN X EMB_DIM
        op_proj = torch.tanh(self.position_project(object_positions))
        
        ### Gather scene state ###
        ### Projected Version ###
        ### BS X SEQ_LEN X EMB_DIM
        ss_proj = torch.tanh(self.scene_project(scene_state))
        
        ### Gather all colors ###
        ### Get embeddings ###
        oc_proj =  self.color_embeddings(object_colors)
        
        ### Gather all shapes ###
        ### Get embeddings ###
        os_proj =  self.shape_embeddings(object_shapes)
        
        ### Gather all materials ###
        ### Get embeddings ###
        om_proj =  self.material_embeddings(object_materials)
        
        ### Gather all sizes ###
        ### Get embeddings ###
        oz_proj =  self.size_embeddings(object_sizes)
        object_related_embeddings = torch.cat([op_proj, oc_proj, os_proj, om_proj, oz_proj],2)
        ### Reproject them to 128 sized-embeddings ###
        ore = self.reproject(object_related_embeddings)
        ### Add CLS token ###
        cls_token = torch.zeros(size=(object_related_embeddings.size(0), 1, self.config['hidden_dim']))
        
        ### Stack all embeddings on the sequence dimension ###
        pre_augmentation_embeddings = torch.cat([cls_token, ore, ss_proj, questions], 1)
        augmentation_embeddings = position_embeddings + type_embeddings
        
        embeddings = pre_augmentation_embeddings + augmentation_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings, mask, object_mask

class MLPClassifierHead(Module):
    def __init__(self, config: dict, use_log_transform=True, mode='soft'):
        super(MLPClassifierHead,self).__init__()
        self.linear_layer  = nn.Linear(config['hidden_dim'], config['num_output_classes'])
        
        if mode == 'arg':
            self.softmax_layer =  lambda x: torch.argmax(x, dim=1, keepdim=True)
        else:
            if use_log_transform:
                self.softmax_layer =  nn.LogSoftmax(dim=1)
            else:
                self.softmax_layer =  nn.Softmax(dim=1)
            
            
        return
    
    def forward(self, input):
        return self.softmax_layer(self.linear_layer(input))

class DeltaFormer(Module):
    def __init__(self, config: dict):
        super(DeltaFormer,self).__init__()
        self.mme = MultiModalEmbedder(config)
        self.be  = BertEncoder(config)
        self.classhead = MLPClassifierHead(config)
        return
    
    def forward(self, **kwargs):
        embeddings, mask, obj_mask = self.mme(**kwargs)
        out = self.be.forward(embeddings, mask, output_all_encoded_layers=False, output_attention_probs=False)
        ### Be returns List ###
        out = out[0]

        ### Get the Item Outputs - Tokens 1-11 ##
        item_output = out[:,1:11]

        ### Filtered Output ###
        filtered_item_output = item_output * obj_mask[:,1:11].unsqueeze(2)

        ### Get the Scene Output - Token 12 - ##
        scene_output = out[:,11]

        ### Get the [CLS] output - Token 0 - ##
        cls_output = out[:,0]
        answer = self.classhead(cls_output)
        return answer, filtered_item_output, scene_output


