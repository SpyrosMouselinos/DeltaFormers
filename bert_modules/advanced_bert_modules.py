import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module as Module

from bert_modules.utils import BertLayerNorm, BertSelfAttention


class PreLNBertBlock(nn.Module):

    def __init__(self, config, dropout=0.1):
        super(PreLNBertBlock, self).__init__()
        self.hidden_dim = config['hidden_dim']
        self.inter_dim = config['inter_dim']
        self.num_heads = config['num_attention_heads']

        self.layer_norm_1 = BertLayerNorm(self.hidden_dim)
        self.attn = BertSelfAttention(config)
        self.layer_norm_2 = BertLayerNorm(self.hidden_dim)

        self.linear = nn.Sequential(
            nn.Linear(self.hidden_dim, self.inter_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.inter_dim, self.hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, attention_mask, use_init_ln=True, linear_head_indicies=None, output_attention_probs=True):
        if use_init_ln:
            inp_x = self.layer_norm_1(x)
        else:
            inp_x = x
        if output_attention_probs:
            outs, att = self.attn(inp_x, attention_mask, linear_head_indicies, output_attention_probs=True)
            if linear_head_indicies is None:
                x = x + outs
                x = x + self.linear(self.layer_norm_2(x))
                return x, att
            else:
                length = x.size(1)
                padded_outs = torch.zeros_like(x, device=x.device)
                for entry in linear_head_indicies:
                    padded_outs += F.pad(input=outs, pad=(0, 0, 2 * entry - length + 1, length - entry - 1),
                                         mode='constant', value=0)
                x = x + padded_outs
                x = x + self.linear(self.layer_norm_2(x))
                return x, att
        else:
            outs = self.attn(inp_x, attention_mask, linear_head_indicies, output_attention_probs=False)
            if linear_head_indicies is None:
                x = x + outs
                x = x + self.linear(self.layer_norm_2(x))
                return x
            else:
                length = x.size(1)
                padded_outs = torch.zeros_like(x, device=x.device)
                for entry in linear_head_indicies:
                    padded_outs += F.pad(input=outs, pad=(0, 0, 2 * entry - length + 1, length - entry - 1),
                                         mode='constant', value=0)
                x = x + padded_outs
                x = x + self.linear(self.layer_norm_2(x))
                return x


class PreLNBertEncoder(Module):
    def __init__(self, config):
        super(PreLNBertEncoder, self).__init__()
        layer = PreLNBertBlock(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config['num_bert_layers'])])
        self.post_ln = BertLayerNorm(config['hidden_dim'])

    def forward(self, hidden_states, attention_mask, linear_head_indicies=None, output_attention_probs=True):
        all_attention_probs = []
        for i, layer_module in enumerate(self.layer):
            use_init_ln = (i != 0)
            hidden_states = layer_module(x=hidden_states, attention_mask=attention_mask,
                                         use_init_ln=use_init_ln, linear_head_indicies=linear_head_indicies,
                                         output_attention_probs=output_attention_probs)
            if output_attention_probs:
                hidden_states, attention_probs = hidden_states
                all_attention_probs.append(attention_probs)

        final_state = self.post_ln(hidden_states)
        if output_attention_probs:
            return final_state, all_attention_probs
        else:
            return final_state

# def_config = {
#     'num_bert_layers': 6,
#     'hidden_dim': 128,
#     'inter_dim': 256,
#     'num_attention_heads': 8,
#     'attention_temperature': 0.5,
# }
#
# device = 'cpu'
# cls = PreLNBert(def_config)
# cls = cls.to(device)
# cls.train()
#
# optim = torch.optim.Adam(cls.parameters())
# from time import time
# import matplotlib.pyplot as plt
#
# squared_times = []
# pseudo_linear_times = []
# metric = torch.nn.MSELoss()
# for i in range(10, 100):
#     optim.zero_grad()
#     hs = torch.ones((32, i, 128)).to(device)
#     am = torch.zeros((32, 1, 1, i)).to(device)
#     real = torch.zeros((32,i, 128)).to(device)
#     tic = time()
#     out = cls.forward(hidden_states=hs, attention_mask=am,
#                      linear_head_indicies=None, output_attention_probs=False)
#     loss = metric(out, real)
#     loss.backward()
#     optim.step()
#     toc = time()
#     squared_times.append(toc - tic)
#
#     optim.zero_grad()
#     # Clear
#     tic = time()
#     out = cls.forward(hidden_states=hs, attention_mask=am,
#                     linear_head_indicies=[i - 1], output_attention_probs=False)
#     loss = metric(out, real)
#     loss.backward()
#     optim.step()
#     toc = time()
#     pseudo_linear_times.append(toc - tic)
#     optim.zero_grad()
#
# plt.figure(figsize=(10, 10))
# plt.title("Time Difference Between Transformer Attention Architectures")
# plt.plot(list(range(11, 100)), squared_times[1:], 'b', label='O(N^2) Attention')
# plt.plot(list(range(11, 100)), pseudo_linear_times[1:], 'r', label='O(k*N) Attention')
# plt.legend()
# plt.ylim()
# plt.savefig('C:\\Users\\Guldan\\Desktop\\CPU_comparison_backprop.png')
# plt.show()
# plt.close()
