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
        self.position_embeddings    = nn.Embedding(config['max_objects_per_scene'] + 1 + config['max_question_tokens_per_scene'], config['hidden_dim'])
        self.token_type_embeddings  = nn.Embedding(config['num_token_types'], config['hidden_dim'], padding_idx=0)
        self.color_embeddings       = nn.Embedding(config['num_colors'], config['embedding_dim'])
        self.shape_embeddings       = nn.Embedding(config['num_shapes'], config['embedding_dim'])
        self.material_embeddings    = nn.Embedding(config['num_materials'], config['embedding_dim'])
        self.size_embeddings        = nn.Embedding(config['num_sizes'], config['embedding_dim'])
        
        self.position_project = nn.Linear(config['num_positions'], config['embedding_dim'])
        self.scene_project    = nn.Linear(config['num_scene_positions'], config['hidden_dim'])
        self.reproject        = nn.Linear(5 * config['embedding_dim'], config['hidden_dim'])
        
        # self.pros_norm        = BertLayerNorm(config['embedding_dim'], eps=1e-12)
        # self.scene_norm       = BertLayerNorm(config['hidden_dim'], eps=1e-12)
        # self.color_norm       = BertLayerNorm(config['embedding_dim'], eps=1e-12)
        # self.shape_norm       = BertLayerNorm(config['embedding_dim'], eps=1e-12)
        # self.material_norm    = BertLayerNorm(config['embedding_dim'], eps=1e-12)
        # self.size_norm        = BertLayerNorm(config['embedding_dim'], eps=1e-12)

        self.pros_norm        = lambda x: x
        self.scene_norm       = lambda x: x
        self.color_norm       = lambda x: x
        self.shape_norm       = lambda x: x
        self.material_norm    = lambda x: x
        self.size_norm        = lambda x: x
        
        self.LayerNorm        = BertLayerNorm(config['hidden_dim'], eps=1e-12)
        #self.dropout          = nn.Dropout(0.1)
        self.dropout          = lambda x: x
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
        op_proj = self.pros_norm(self.position_project(object_positions))
        
        ### Gather scene state ###
        ### Projected Version ###
        ### BS X SEQ_LEN X EMB_DIM
        ss_proj = self.scene_norm(self.scene_project(scene_state))
        
        ### Gather all colors ###
        ### Get embeddings ###
        oc_proj =  self.color_norm(self.color_embeddings(object_colors))
        
        ### Gather all shapes ###
        ### Get embeddings ###
        os_proj =  self.shape_norm(self.shape_embeddings(object_shapes))
        
        ### Gather all materials ###
        ### Get embeddings ###
        om_proj =  self.material_norm(self.material_embeddings(object_materials))
        
        ### Gather all sizes ###
        ### Get embeddings ###
        oz_proj =  self.size_norm(self.size_embeddings(object_sizes))
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

class MLPClassifierHead(Module):
    def __init__(self, config: dict, use_log_transform=False, mode='raw'):
        super(MLPClassifierHead,self).__init__()
        self.linear_layer_1  = nn.Linear(config['hidden_dim'], config['hidden_dim'])
        self.linear_layer_2  = nn.Linear(config['hidden_dim'], config['num_output_classes'])
        
        if mode == 'arg':
            self.softmax_layer =  lambda x: torch.argmax(x, dim=1, keepdim=True)
        elif mode == 'soft':
            if use_log_transform:
                self.softmax_layer =  nn.LogSoftmax(dim=1)
            else:
                self.softmax_layer =  nn.Softmax(dim=1)
        elif mode == 'raw':
            self.softmax_layer =  lambda x: x
        else:
            raise NotImplementedError(f"Mode: {mode} not implemented in MLPClassifierHead Module...")
        return
    
    def forward(self, input):
        input = nn.ReLU()(self.linear_layer_1(input))
        return self.softmax_layer(self.linear_layer_2(input))
    
class ConcatClassifierHead(Module):
    def __init__(self, config: dict):
        super(ConcatClassifierHead,self).__init__()
        self.linear_layer_1  = nn.Linear(config['max_objects_per_scene'] * config['hidden_dim'], config['hidden_dim'])
        self.linear_layer_2  = nn.Linear(config['hidden_dim'], config['num_output_classes'])
    
    def forward(self, input_set):
        flat_set = input_set.view(input_set.size(0), -1)
        flat_set = nn.ReLU()(self.linear_layer_1(flat_set))
        return self.linear_layer_2(flat_set)
    
    
class DeltaFormer(Module):
    def __init__(self, config: dict):
        super(DeltaFormer,self).__init__()
        self.mme = MultiModalEmbedder(config)
        self.be  = BertEncoder(config)
        self.classhead = MLPClassifierHead(config)
        self.concathead = ConcatClassifierHead(config)
        return
    
    def forward(self, **kwargs):
        embeddings, mask, obj_mask = self.mme(**kwargs)
        out = self.be.forward(embeddings, mask, output_all_encoded_layers=False, output_attention_probs=False)
        ### Be returns List ###
        out = out[0]

        ### Get the Item Outputs - Tokens 0-10 ##
        item_output = out[:,0:10]

        ### Filtered Output ###
        filtered_item_output = item_output * obj_mask[:,0:10].unsqueeze(2)

        ### Get the Scene Output - Token 11 - ##
        scene_output = out[:,10]

        ### Get the [START] output - Token 12 - ##
        cls_output = out[:,11]
        answer = self.classhead(cls_output)
        filtered_item_output = self.concathead(filtered_item_output)
        return answer, filtered_item_output, scene_output


