import copy

import torch.nn as nn
from torch.nn import Module as Module

from bert_modules.utils import BertLayerNorm, BertSelfAttention, gelu


class BertSelfOutput(Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config['hidden_dim'], config['hidden_dim'])
        self.LayerNorm = BertLayerNorm(config['hidden_dim'], eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask, output_attention_probs=False):
        self_output = self.self(input_tensor, attention_mask, output_attention_probs=output_attention_probs)
        if output_attention_probs:
            self_output, attention_probs = self_output
        attention_output = self.output(self_output, input_tensor)
        if output_attention_probs:
            return attention_output, attention_probs
        return attention_output


class BertIntermediate(Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config['hidden_dim'], config['inter_dim'])
        self.intermediate_act_fn = gelu

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config['inter_dim'], config['hidden_dim'])
        self.LayerNorm = BertLayerNorm(config['hidden_dim'], eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, output_attention_probs=False):
        attention_output = self.attention(hidden_states, attention_mask, output_attention_probs=output_attention_probs)
        if output_attention_probs:
            attention_output, attention_probs = attention_output
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        if output_attention_probs:
            return layer_output, attention_probs
        else:
            return layer_output


class BertEncoder(Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config['num_bert_layers'])])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=False, output_attention_probs=True):
        all_encoder_layers = []
        all_attention_probs = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask, output_attention_probs=output_attention_probs)
            if output_attention_probs:
                hidden_states, attention_probs = hidden_states
                all_attention_probs.append(attention_probs)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        if output_attention_probs:
            return all_encoder_layers, all_attention_probs
        else:
            return all_encoder_layers
