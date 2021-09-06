import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def str_to_function(s):
    if '[' not in s:
        return {
            'function': s,
            'value_inputs': [],
        }
    name, value_str = s.replace(']', '').split('[')
    return {
        'function': name,
        'value_inputs': value_str.split(','),
    }


def get_num_inputs(f):
    # This is a litle hacky; it would be better to look up from metadata.json
    if type(f) is str:
        f = str_to_function(f)
    name = f['function']
    if name == 'scene':
        return 0
    if 'equal' in name or name in ['union', 'intersect', 'less_than', 'greater_than']:
        return 2
    return 1


def function_to_str(f):
    value_str = ''
    if f['value_inputs']:
        value_str = '[%s]' % ','.join(f['value_inputs'])
    return '%s%s' % (f['function'], value_str)


def build_stem(feature_dim, module_dim, num_layers=2, with_batchnorm=True):
    layers = []
    prev_dim = feature_dim
    for i in range(num_layers):
        layers.append(nn.Conv2d(prev_dim, module_dim, kernel_size=(3, 3), padding=(1, 1)))
        if with_batchnorm:
            layers.append(nn.BatchNorm2d(module_dim))
        layers.append(nn.ReLU(inplace=True))
        prev_dim = module_dim
    return nn.Sequential(*layers)


def build_classifier(module_C, module_H, module_W, num_answers,
                     fc_dims=[], proj_dim=None, downsample='maxpool2',
                     with_batchnorm=True, dropout=0):
    layers = []
    prev_dim = module_C * module_H * module_W
    if proj_dim is not None and proj_dim > 0:
        layers.append(nn.Conv2d(module_C, proj_dim, kernel_size=(1, 1)))
        if with_batchnorm:
            layers.append(nn.BatchNorm2d(proj_dim))
        layers.append(nn.ReLU(inplace=True))
        prev_dim = proj_dim * module_H * module_W
    if downsample == 'maxpool2':
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        prev_dim //= 4
    elif downsample == 'maxpool4':
        layers.append(nn.MaxPool2d(kernel_size=4, stride=4))
        prev_dim //= 16
    layers.append(Flatten())
    for next_dim in fc_dims:
        layers.append(nn.Linear(prev_dim, next_dim))
        if with_batchnorm:
            layers.append(nn.BatchNorm1d(next_dim))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        prev_dim = next_dim
    layers.append(nn.Linear(prev_dim, num_answers))
    return nn.Sequential(*layers)


def logical_or(x, y):
    return (x + y).clamp_(0, 1)


def logical_not(x):
    return x == 0


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim=None, with_residual=True, with_batchnorm=True):
        if out_dim is None:
            out_dim = in_dim
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=(3, 3), padding=(1, 1))
        self.with_batchnorm = with_batchnorm
        if with_batchnorm:
            self.bn1 = nn.BatchNorm2d(out_dim)
            self.bn2 = nn.BatchNorm2d(out_dim)
        self.with_residual = with_residual
        if in_dim == out_dim or not with_residual:
            self.proj = None
        else:
            self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=(1, 1))

    def forward(self, x):
        if self.with_batchnorm:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
        else:
            out = self.conv2(F.relu(self.conv1(x)))
        res = x if self.proj is None else self.proj(x)
        if self.with_residual:
            out = F.relu(res + out)
        else:
            out = F.relu(out)
        return out


class Seq2Seq(nn.Module):
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
                 encoder_embed=None
                 ):
        super(Seq2Seq, self).__init__()
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
        self.multinomial_outputs = None

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
                if x_cpu.data[i, t] != self.NULL and x_cpu.data[i, t + 1] == self.NULL:
                    idx[i] = t
                    break
        idx = idx.type_as(x.data)
        x[x.data == self.NULL] = replace
        return x, Variable(idx)

    def encoder(self, x):
        V_in, V_out, D, H, L, N, T_in, T_out = self.get_dims(x=x)
        x, idx = self.before_rnn(x)
        embed = self.encoder_embed(x)
        h0 = Variable(torch.zeros(L, N, H).type_as(embed.data))
        c0 = Variable(torch.zeros(L, N, H).type_as(embed.data))

        out, _ = self.encoder_rnn(embed, (h0, c0))

        # Pull out the hidden state for the last non-null value in each input
        idx = idx.view(N, 1, 1).expand(N, 1, H)
        return out.gather(1, idx).view(N, H)

    def decoder(self, encoded, y, h0=None, c0=None):
        V_in, V_out, D, H, L, N, T_in, T_out = self.get_dims(y=y)

        if T_out > 1:
            y, _ = self.before_rnn(y)
        y_embed = self.decoder_embed(y)
        encoded_repeat = encoded.view(N, 1, H).expand(N, T_out, H)
        rnn_input = torch.cat([encoded_repeat, y_embed], 2)
        if h0 is None:
            h0 = Variable(torch.zeros(L, N, H).type_as(encoded.data))
        if c0 is None:
            c0 = Variable(torch.zeros(L, N, H).type_as(encoded.data))
        rnn_output, (ht, ct) = self.decoder_rnn(rnn_input, (h0, c0))

        rnn_output_2d = rnn_output.contiguous().view(N * T_out, H)
        output_logprobs = self.decoder_linear(rnn_output_2d).view(N, T_out, V_out)

        return output_logprobs, ht, ct

    def compute_loss(self, output_logprobs, y):
        """
        Compute loss. We assume that the first element of the output sequence y is
        a start token, and that each element of y is left-aligned and right-padded
        with self.NULL out to T_out. We want the output_logprobs to predict the
        sequence y, shifted by one timestep so that y[0] is fed to the network and
        then y[1] is predicted. We also don't want to compute loss for padded
        timesteps.

        Inputs:
        - output_logprobs: Variable of shape (N, T_out, V_out)
        - y: LongTensor Variable of shape (N, T_out)
        """
        self.multinomial_outputs = None
        V_in, V_out, D, H, L, N, T_in, T_out = self.get_dims(y=y)
        mask = y.data != self.NULL
        y_mask = Variable(torch.Tensor(N, T_out).fill_(0).type_as(mask))
        y_mask[:, 1:] = mask[:, 1:]
        y_masked = y[y_mask]
        out_mask = Variable(torch.Tensor(N, T_out).fill_(0).type_as(mask))
        out_mask[:, :-1] = mask[:, 1:]
        out_mask = out_mask.view(N, T_out, 1).expand(N, T_out, V_out)
        out_masked = output_logprobs[out_mask].view(-1, V_out)
        loss = F.cross_entropy(out_masked, y_masked)
        return loss

    def forward(self, x, y):
        encoded = self.encoder(x)
        output_logprobs, _, _ = self.decoder(encoded, y)
        loss = self.compute_loss(output_logprobs, y)
        return loss

    def sample(self, x, max_length=50):
        # TODO: Handle sampling for minibatch inputs
        # TODO: Beam search?
        self.multinomial_outputs = None
        assert x.size(0) == 1, "Sampling minibatches not implemented"
        encoded = self.encoder(x)
        y = [self.START]
        h0, c0 = None, None
        while True:
            cur_y = Variable(torch.LongTensor([y[-1]]).type_as(x.data).view(1, 1))
            logprobs, h0, c0 = self.decoder(encoded, cur_y, h0=h0, c0=c0)
            _, next_y = logprobs.data.max(2)
            y.append(next_y[0, 0, 0])
            if len(y) >= max_length or y[-1] == self.END:
                break
        return y

    def reinforce_sample(self, x, max_length=30, temperature=1.0, argmax=False):
        N, T = x.size(0), max_length
        encoded = self.encoder(x)
        y = torch.LongTensor(N, T).fill_(self.NULL)
        done = torch.ByteTensor(N).fill_(0)
        cur_input = Variable(x.data.new(N, 1).fill_(self.START))
        h, c = None, None
        self.multinomial_outputs = []
        self.multinomial_probs = []
        for t in range(T):
            # logprobs is N x 1 x V
            logprobs, h, c = self.decoder(encoded, cur_input, h0=h, c0=c)
            logprobs = logprobs / temperature
            probs = F.softmax(logprobs.view(N, -1), dim=1)  # Now N x V
            _, cur_output = probs.max(1)
            self.multinomial_outputs.append(cur_output)
            self.multinomial_probs.append(probs)
            cur_output_data = cur_output.data.cpu()
            not_done = logical_not(done)
            y[:, t][not_done] = cur_output_data[not_done]
            done = logical_or(done, cur_output_data.cpu() == self.END)
            cur_input = cur_output.view(N, 1).expand(N, 1)
            if done.sum() == N:
                break
        return Variable(y.type_as(x.data))


class ModuleNet(nn.Module):
    def __init__(self, vocab, feature_dim=(1024, 14, 14),
                 stem_num_layers=2,
                 stem_batchnorm=False,
                 module_dim=128,
                 module_residual=True,
                 module_batchnorm=False,
                 classifier_proj_dim=512,
                 classifier_downsample='maxpool2',
                 classifier_fc_layers=(1024,),
                 classifier_batchnorm=False,
                 classifier_dropout=0,
                 verbose=True):
        super(ModuleNet, self).__init__()

        self.stem = build_stem(feature_dim[0], module_dim,
                               num_layers=stem_num_layers,
                               with_batchnorm=stem_batchnorm)
        if verbose:
            print('Here is my stem:')
            print(self.stem)

        num_answers = len(vocab['answer_idx_to_token'])
        module_H, module_W = feature_dim[1], feature_dim[2]
        self.classifier = build_classifier(module_dim, module_H, module_W, num_answers,
                                           classifier_fc_layers,
                                           classifier_proj_dim,
                                           classifier_downsample,
                                           with_batchnorm=classifier_batchnorm,
                                           dropout=classifier_dropout)
        if verbose:
            print('Here is my classifier:')
            print(self.classifier)
        self.stem_times = []
        self.module_times = []
        self.classifier_times = []
        self.timing = False

        self.function_modules = {}
        self.function_modules_num_inputs = {}
        self.vocab = vocab
        for fn_str in vocab['program_token_to_idx']:
            num_inputs = get_num_inputs(fn_str)
            self.function_modules_num_inputs[fn_str] = num_inputs
            if fn_str == 'scene' or num_inputs == 1:
                mod = ResidualBlock(module_dim,
                                    with_residual=module_residual,
                                    with_batchnorm=module_batchnorm)
            elif num_inputs == 2:
                mod = ConcatBlock(module_dim,
                                  with_residual=module_residual,
                                  with_batchnorm=module_batchnorm)
            self.add_module(fn_str, mod)
            self.function_modules[fn_str] = mod

        self.save_module_outputs = False

    def expand_answer_vocab(self, answer_to_idx, std=0.01, init_b=-50):
        final_linear_key = str(len(self.classifier._modules) - 1)
        final_linear = self.classifier._modules[final_linear_key]

        old_weight = final_linear.weight.data
        old_bias = final_linear.bias.data
        old_N, D = old_weight.size()
        new_N = 1 + max(answer_to_idx.values())
        new_weight = old_weight.new(new_N, D).normal_().mul_(std)
        new_bias = old_bias.new(new_N).fill_(init_b)
        new_weight[:old_N].copy_(old_weight)
        new_bias[:old_N].copy_(old_bias)

        final_linear.weight.data = new_weight
        final_linear.bias.data = new_bias

    def _forward_modules_json(self, feats, program):
        def gen_hook(i, j):
            def hook(grad):
                self.all_module_grad_outputs[i][j] = grad.data.cpu().clone()

            return hook

        self.all_module_outputs = []
        self.all_module_grad_outputs = []
        # We can't easily handle minibatching of modules, so just do a loop
        N = feats.size(0)
        final_module_outputs = []
        for i in range(N):
            if self.save_module_outputs:
                self.all_module_outputs.append([])
                self.all_module_grad_outputs.append([None] * len(program[i]))
            module_outputs = []
            for j, f in enumerate(program[i]):
                f_str = function_to_str(f)
                module = self.function_modules[f_str]
                if f_str == 'scene':
                    module_inputs = [feats[i:i + 1]]
                else:
                    module_inputs = [module_outputs[j] for j in f['inputs']]
                module_outputs.append(module(*module_inputs))
                if self.save_module_outputs:
                    self.all_module_outputs[-1].append(module_outputs[-1].data.cpu().clone())
                    module_outputs[-1].register_hook(gen_hook(i, j))
            final_module_outputs.append(module_outputs[-1])
        final_module_outputs = torch.cat(final_module_outputs, 0)
        return final_module_outputs

    def _forward_modules_ints_helper(self, feats, program, i, j):
        used_fn_j = True
        if j < program.size(1):
            fn_idx = int(program.data[i, j].cpu().numpy())
            fn_str = self.vocab['program_idx_to_token'][fn_idx]
        else:
            used_fn_j = False
            fn_str = 'scene'
        if fn_str == '<NULL>':
            used_fn_j = False
            fn_str = 'scene'
        elif fn_str == '<START>':
            used_fn_j = False
            return self._forward_modules_ints_helper(feats, program, i, j + 1)
        if used_fn_j:
            self.used_fns[i, j] = 1
        j += 1
        module = self.function_modules[fn_str]
        if fn_str == 'scene':
            module_inputs = [feats[i:i + 1]]
        else:
            num_inputs = self.function_modules_num_inputs[fn_str]
            module_inputs = []
            while len(module_inputs) < num_inputs:
                cur_input, j = self._forward_modules_ints_helper(feats, program, i, j)
                module_inputs.append(cur_input)
        module_output = module(*module_inputs)
        return module_output, j

    def _forward_modules_ints(self, feats, program):
        """
        feats: FloatTensor of shape (N, C, H, W) giving features for each image
        program: LongTensor of shape (N, L) giving a prefix-encoded program for
          each image.
        """
        N = feats.size(0)
        final_module_outputs = []
        self.used_fns = torch.Tensor(program.size()).fill_(0)
        for i in range(N):
            cur_output, _ = self._forward_modules_ints_helper(feats, program, i, 0)
            final_module_outputs.append(cur_output)
        self.used_fns = self.used_fns.type_as(program.data).float()
        final_module_outputs = torch.cat(final_module_outputs, 0)
        return final_module_outputs

    def forward(self, x, program):
        N = x.size(0)
        assert N == len(program)

        feats = self.stem(x)

        if type(program) is list or type(program) is tuple:
            final_module_outputs = self._forward_modules_json(feats, program)
        elif type(program) is torch.Tensor and program.dim() == 2:
            final_module_outputs = self._forward_modules_ints(feats, program)
        else:
            raise ValueError('Unrecognized program format')
        out = self.classifier(final_module_outputs)
        return out


class ConcatBlock(nn.Module):
    def __init__(self, dim, with_residual=True, with_batchnorm=True):
        super(ConcatBlock, self).__init__()
        self.proj = nn.Conv2d(2 * dim, dim, kernel_size=(1, 1), padding=(0, 0))
        self.res_block = ResidualBlock(dim, with_residual=with_residual,
                                       with_batchnorm=with_batchnorm)

    def forward(self, x, y):
        out = torch.cat([x, y], 1)  # Concatentate along depth
        out = F.relu(self.proj(out))
        out = self.res_block(out)
        return out


class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.size(0), -1)
