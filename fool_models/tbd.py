import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def logical_or(x, y):
    return (x + y).clamp_(0, 1)


def logical_not(x):
    return x == 0


class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.size(0), -1)


class TbDNet(nn.Module):
    def __init__(self,
                 vocab,
                 feature_dim=(512, 28, 28),
                 module_dim=128,
                 cls_proj_dim=512,
                 fc_dim=1024):
        super().__init__()
        # Why in God's name would you change the order?
        self.translate_codes = {0: 15,
                           1: 16,
                           2: 18,
                           3: 20,
                           4: 21,
                           5: 25,
                           6: 26,
                           7: 30,
                           8: 17,
                           9: 19,
                           10: 29,
                           11: 22,
                           12: 28,
                           13: 23,
                           14: 27,
                           15: 24,
                           16: 31,
                           17: 4,
                           18: 5,
                           19: 6,
                           20: 7,
                           21: 8,
                           22: 9,
                           23: 10,
                           24: 11,
                           25: 12,
                           26: 13,
                           27: 14}
        self.stem = nn.Sequential(nn.Conv2d(feature_dim[0], module_dim, kernel_size=(3, 3), padding=1),
                                  nn.ReLU(),
                                  nn.Conv2d(module_dim, module_dim, kernel_size=(3, 3), padding=1),
                                  nn.ReLU()
                                  )

        module_rows, module_cols = feature_dim[1], feature_dim[2]
        self.classifier = nn.Sequential(nn.Conv2d(module_dim, cls_proj_dim, kernel_size=(1, 1)),
                                        nn.ReLU(inplace=True),
                                        nn.MaxPool2d(kernel_size=2, stride=2),
                                        Flatten(),
                                        nn.Linear(cls_proj_dim * module_rows * module_cols // 4,
                                                  fc_dim),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(fc_dim, 28)  # note no softmax here
                                        )

        self.function_modules = {}  # holds our modules
        self.vocab = vocab
        # go through the vocab and add all the modules to our model
        for module_name in vocab['program_token_to_idx']:
            if module_name in ['<NULL>', '<START>', '<END>', '<UNK>', 'unique']:
                continue  # we don't need modules for the placeholders

            # figure out which module we want we use
            if module_name == 'scene':
                # scene is just a flag that indicates the start of a new line of reasoning
                # we set `module` to `None` because we still need the flag 'scene' in forward()
                module = None
            elif module_name == 'intersect':
                module = AndModule()
            elif module_name == 'union':
                module = OrModule()
            elif 'equal' in module_name or module_name in {'less_than', 'greater_than'}:
                module = ComparisonModule(module_dim)
            elif 'query' in module_name or module_name in {'exist', 'count'}:
                module = QueryModule(module_dim)
            elif 'relate' in module_name:
                module = RelateModule(module_dim)
            elif 'same' in module_name:
                module = SameModule(module_dim)
            else:
                module = AttentionModule(module_dim)

            # add the module to our dictionary and register its parameters so it can learn
            self.function_modules[module_name] = module
            self.add_module(module_name, module)
        # this is used as input to the first AttentionModule in each program
        ones = torch.ones(1, 1, module_rows, module_cols)
        self.ones_var = ones.cuda() if torch.cuda.is_available() else ones
        self._attention_sum = 0

    @property
    def attention_sum(self):
        return self._attention_sum

    def forward(self, feats, programs):
        batch_size = feats.size(0)
        assert batch_size == len(programs)

        feat_input_volume = self.stem(feats)
        final_module_outputs = []
        self._attention_sum = 0
        for n in range(batch_size):
            feat_input = feat_input_volume[n:n + 1]
            output = feat_input
            saved_output = None
            for i in reversed(programs.data[n].cpu().numpy()):
                module_type = self.vocab['program_idx_to_token'][i]
                if module_type in {'<NULL>', '<START>', '<END>', '<UNK>', 'unique'}:
                    continue  # the above are no-ops in our model

                module = self.function_modules[module_type]
                if module_type == 'scene':
                    # store the previous output; it will be needed later
                    # scene is just a flag, performing no computation
                    saved_output = output
                    output = self.ones_var
                    continue

                if 'equal' in module_type or module_type in {'intersect', 'union', 'less_than',
                                                             'greater_than'}:
                    output = module(output, saved_output)  # these modules take two feature maps
                else:
                    # these modules take extracted image features and a previous attention
                    output = module(feat_input, output)

                if any(t in module_type for t in ['filter', 'relate', 'same']):
                    self._attention_sum += output.sum()

            final_module_outputs.append(output)

        final_module_outputs = torch.cat(final_module_outputs, 0)
        scores = self.classifier(final_module_outputs)
        return scores


    def forward_and_return_intermediates(self, program_var, feats_var):
        intermediaries = []
        # the logic here is the same as self.forward()
        scene_input = self.stem(feats_var)
        output = scene_input
        saved_output = None
        for i in reversed(program_var.data.cpu().numpy()[0]):
            module_type = self.vocab['program_idx_to_token'][i]
            if module_type in {'<NULL>', '<START>', '<END>', '<UNK>', 'unique'}:
                continue

            module = self.function_modules[module_type]
            if module_type == 'scene':
                saved_output = output
                output = self.ones_var
                intermediaries.append(None)  # indicates a break/start of a new logic chain
                continue

            if 'equal' in module_type or module_type in {'intersect', 'union', 'less_than',
                                                         'greater_than'}:
                output = module(output, saved_output)
            else:
                output = module(scene_input, output)

            if module_type in {'intersect', 'union'}:
                intermediaries.append(None)  # this is the start of a new logic chain

            if module_type in {'intersect', 'union'} or any(s in module_type for s in ['same',
                                                                                       'filter',
                                                                                       'relate']):
                intermediaries.append((module_type, output.data.cpu().numpy().squeeze()))

        _, pred = self.classifier(output).max(1)
        return self.vocab['answer_idx_to_token'][pred.item()], intermediaries


class AndModule(nn.Module):
    def forward(self, attn1, attn2):
        out = torch.min(attn1, attn2)
        return out


class OrModule(nn.Module):
    def forward(self, attn1, attn2):
        out = torch.max(attn1, attn2)
        return out


class AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv2d(dim, 1, kernel_size=(1, 1), padding=0)
        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        torch.nn.init.kaiming_normal_(self.conv3.weight)
        self.dim = dim

    def forward(self, feats, attn):
        attended_feats = torch.mul(feats, attn.repeat(1, self.dim, 1, 1))
        out = F.relu(self.conv1(attended_feats))
        out = F.relu(self.conv2(out))
        out = torch.sigmoid(self.conv3(out))
        return out


class QueryModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=(3, 3), padding=1)
        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        self.dim = dim

    def forward(self, feats, attn):
        attended_feats = torch.mul(feats, attn.repeat(1, self.dim, 1, 1))
        out = F.relu(self.conv1(attended_feats))
        out = F.relu(self.conv2(out))
        return out


class RelateModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=(3, 3), padding=1, dilation=(1, 1))  # receptive field 3
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=(3, 3), padding=2, dilation=(2, 2))  # 7
        self.conv3 = nn.Conv2d(dim, dim, kernel_size=(3, 3), padding=4, dilation=(4, 4))  # 15
        self.conv4 = nn.Conv2d(dim, dim, kernel_size=(3, 3), padding=8, dilation=(8, 8))  # 31 -- full image
        self.conv5 = nn.Conv2d(dim, dim, kernel_size=(3, 3), padding=1, dilation=(1, 1))
        self.conv6 = nn.Conv2d(dim, 1, kernel_size=(1, 1), padding=0)
        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        torch.nn.init.kaiming_normal_(self.conv3.weight)
        torch.nn.init.kaiming_normal_(self.conv4.weight)
        torch.nn.init.kaiming_normal_(self.conv5.weight)
        torch.nn.init.kaiming_normal_(self.conv6.weight)
        self.dim = dim

    def forward(self, feats, attn):
        feats = torch.mul(feats, attn.repeat(1, self.dim, 1, 1))
        out = F.relu(self.conv1(feats))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = F.relu(self.conv5(out))
        out = torch.sigmoid(self.conv6(out))
        return out


class SameModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim + 1, 1, kernel_size=(1, 1))
        torch.nn.init.kaiming_normal_(self.conv.weight)
        self.dim = dim

    def forward(self, feats, attn):
        size = attn.size()[2]
        the_max, the_idx = F.max_pool2d(attn, size, return_indices=True)
        attended_feats = feats.index_select(2, torch.div(the_idx[0, 0, 0, 0], size, rounding_mode='floor'))
        attended_feats = attended_feats.index_select(3, the_idx[0, 0, 0, 0] % size)
        x = torch.mul(feats, attended_feats.repeat(1, 1, size, size))
        x = torch.cat([x, attn], dim=1)
        out = torch.sigmoid(self.conv(x))
        return out


class ComparisonModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.projection = nn.Conv2d(2 * dim, dim, kernel_size=(1, 1), padding=0)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=(3, 3), padding=1)
        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)

    def forward(self, in1, in2):
        out = torch.cat([in1, in2], 1)
        out = F.relu(self.projection(out))
        out = F.relu(self.conv1(out))
        out = F.relu(self.conv2(out))
        return out


class _Seq2Seq(nn.Module):
    def __init__(self,
                 encoder_vocab_size=100,
                 decoder_vocab_size=100,
                 wordvec_dim=300,
                 hidden_dim=256,
                 rnn_num_layers=2,
                 rnn_dropout=0,
                 null_token=0,
                 start_token=1,
                 end_token=2,
                 encoder_embed=None):
        super().__init__()
        self.encoder_embed = nn.Embedding(encoder_vocab_size, wordvec_dim)
        self.encoder_rnn = nn.LSTM(wordvec_dim, hidden_dim, rnn_num_layers,
                                   dropout=rnn_dropout, batch_first=True)
        self.decoder_embed = nn.Embedding(decoder_vocab_size, wordvec_dim)
        self.decoder_rnn = nn.LSTM(wordvec_dim + hidden_dim, hidden_dim, rnn_num_layers,
                                   dropout=rnn_dropout, batch_first=True)
        self.decoder_linear = nn.Linear(hidden_dim, decoder_vocab_size)
        self.NULL = null_token
        self.START = start_token
        self.END = end_token

    def get_dims(self, x=None, y=None):
        V_in = self.encoder_embed.num_embeddings
        V_out = self.decoder_embed.num_embeddings
        D = self.encoder_embed.embedding_dim
        H = self.encoder_rnn.hidden_size
        L = self.encoder_rnn.num_layers

        N = x.size(0) if x is not None else None
        N = y.size(0) if N is None and y is not None else N
        T_in = x.size(1) if x is not None else None
        T_out = y.size(1) if y is not None else None
        return V_in, V_out, D, H, L, N, T_in, T_out

    def before_rnn(self, x, replace=0):
        N, T = x.size()
        idx = torch.LongTensor(N).fill_(T - 1)
        x_cpu = x.cpu()
        for i in range(N):
            for t in range(T - 1):
                if x_cpu[i, t] != self.NULL and x_cpu[i, t + 1] == self.NULL:
                    idx[i] = t
                    break
        idx = idx.type_as(x)
        x[x == self.NULL] = replace
        return x, idx

    def encoder(self, x):
        V_in, V_out, D, H, L, N, T_in, T_out = self.get_dims(x=x)
        x, idx = self.before_rnn(x)
        embed = self.encoder_embed(x)
        h0 = torch.zeros(L, N, H).type_as(embed)
        c0 = torch.zeros(L, N, H).type_as(embed)
        out, _ = self.encoder_rnn(embed, (h0, c0))
        idx = idx.view(N, 1, 1).expand(N, 1, H)
        return out.gather(1, idx).view(N, H)

    def decoder(self, encoded, y, h0=None, c0=None):
        V_in, V_out, D, H, L, N, T_in, T_out = self.get_dims(y=y)

        if T_out > 1:
            y, _ = self.before_rnn(y)
        y_embed = self.decoder_embed(y)
        encoded_repeat = encoded.view(N, 1, H)
        encoded_repeat = encoded_repeat.expand(N, T_out, H)
        rnn_input = torch.cat([encoded_repeat, y_embed], 2)
        if h0 is None:
            h0 = torch.zeros(L, N, H).type_as(encoded)
        if c0 is None:
            c0 = torch.zeros(L, N, H).type_as(encoded)
        rnn_output, (ht, ct) = self.decoder_rnn(rnn_input, (h0, c0))

        rnn_output_2d = rnn_output.contiguous().view(N * T_out, H)
        output_logprobs = self.decoder_linear(rnn_output_2d).view(N, T_out, V_out)

        return output_logprobs, ht, ct

    def reinforce_sample(self, x, max_length=30, temperature=1, argmax=True):
        N, T = x.size(0), max_length
        encoded = self.encoder(x)
        y = torch.LongTensor(N, T).fill_(self.NULL)
        done = torch.ByteTensor(N).fill_(0)
        cur_input = x.new(N, 1).fill_(self.START)
        h, c = None, None
        for t in range(T):
            # logprobs is N x 1 x V
            logprobs, h, c = self.decoder(encoded, cur_input, h0=h, c0=c)
            probs = F.softmax(logprobs.view(N, -1), dim=1)  # Now N x V
            _, cur_output = probs.max(1)
            cur_output = cur_output.unsqueeze(1)
            cur_output_data = cur_output.cpu()
            not_done = logical_not(done)
            y[:, t][not_done] = cur_output_data[not_done][0]
            done = logical_or(done, cur_output_data.cpu().squeeze() == self.END)
            cur_input = cur_output
            if done.sum() == N:
                break
        return y.type_as(x)
