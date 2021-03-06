import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def init_rnn(rnn_type, hidden_dim1, hidden_dim2, rnn_num_layers,
             dropout=0, bidirectional=False):
    if rnn_type == 'gru':
        return nn.GRU(hidden_dim1, hidden_dim2, rnn_num_layers, dropout=dropout,
                      batch_first=True, bidirectional=bidirectional)
    elif rnn_type == 'lstm':
        return nn.LSTM(hidden_dim1, hidden_dim2, rnn_num_layers, dropout=dropout,
                       batch_first=True, bidirectional=bidirectional)
    elif rnn_type == 'linear':
        return None
    else:
        print('RNN type ' + str(rnn_type) + ' not yet implemented.')
        raise (NotImplementedError)


def logical_or(x, y):
    return (x + y).clamp_(0, 1)


def logical_not(x):
    return x == 0


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
        # TODO: Use PackedSequence instead of manually plucking out the last
        # non-NULL entry of each sequence; it is cleaner and more efficient.
        N, T = x.size()
        idx = torch.LongTensor(N).fill_(T - 1)

        # Find the last non-null element in each sequence. Is there a clean
        # way to do this?
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
            probs = F.softmax(logprobs.view(N, -1))  # Now N x V
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


class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.size(0), -1)


def build_stem_film(feature_dim, module_dim, num_layers=2, with_batchnorm=True,
                    kernel_size=3, stride=1, padding=None):
    layers = []
    prev_dim = feature_dim
    if padding is None:  # Calculate default padding when None provided
        if kernel_size % 2 == 0:
            raise (NotImplementedError)
        padding = kernel_size // 2
    for i in range(num_layers):
        layers.append(nn.Conv2d(prev_dim, module_dim, kernel_size=(kernel_size, kernel_size), stride=(stride, stride),
                                padding=padding))
        if with_batchnorm:
            layers.append(nn.BatchNorm2d(module_dim))
        layers.append(nn.ReLU(inplace=True))
        prev_dim = module_dim
    return nn.Sequential(*layers)


def build_classifier_film(module_C, module_H, module_W, num_answers,
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
    if 'maxpool' in downsample or 'avgpool' in downsample:
        pool = nn.MaxPool2d if 'maxpool' in downsample else nn.AvgPool2d
        if 'full' in downsample:
            if module_H != module_W:
                assert (NotImplementedError)
            pool_size = module_H
        else:
            pool_size = int(downsample[-1])
        # Note: Potentially sub-optimal padding for non-perfectly aligned pooling
        padding = 0 if ((module_H % pool_size == 0) and (module_W % pool_size == 0)) else 1
        layers.append(pool(kernel_size=pool_size, stride=pool_size, padding=padding))
        prev_dim = proj_dim * math.ceil(module_H / pool_size) * math.ceil(module_W / pool_size)
    if downsample == 'aggressive':
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        layers.append(nn.AvgPool2d(kernel_size=module_H // 2, stride=module_W // 2))
        prev_dim = proj_dim
        fc_dims = []  # No FC layers here
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


class FiLM(nn.Module):
    """
    A Feature-wise Linear Modulation Layer from
    'FiLM: Visual Reasoning with a General Conditioning Layer'
    """

    def forward(self, x, gammas, betas):
        gammas = gammas.unsqueeze(2).unsqueeze(3).expand_as(x)
        betas = betas.unsqueeze(2).unsqueeze(3).expand_as(x)
        return (gammas * x) + betas


class FiLMedNet(nn.Module):
    def __init__(self, vocab, feature_dim=(1024, 14, 14),
                 stem_num_layers=2,
                 stem_batchnorm=False,
                 stem_kernel_size=3,
                 stem_stride=1,
                 stem_padding=None,
                 num_modules=4,
                 module_num_layers=1,
                 module_dim=128,
                 module_residual=True,
                 module_batchnorm=False,
                 module_batchnorm_affine=False,
                 module_dropout=0,
                 module_input_proj=1,
                 module_kernel_size=3,
                 classifier_proj_dim=512,
                 classifier_downsample='maxpool2',
                 classifier_fc_layers=(1024,),
                 classifier_batchnorm=False,
                 classifier_dropout=0,
                 condition_method='bn-film',
                 condition_pattern=[],
                 use_gamma=True,
                 use_beta=True,
                 use_coords=1,
                 debug_every=float('inf'),
                 print_verbose_every=float('inf'),
                 verbose=True,
                 ):
        super(FiLMedNet, self).__init__()

        num_answers = len(vocab['answer_idx_to_token'])

        self.stem_times = []
        self.module_times = []
        self.classifier_times = []
        self.timing = False

        self.num_modules = num_modules
        self.module_num_layers = module_num_layers
        self.module_batchnorm = module_batchnorm
        self.module_dim = module_dim
        self.condition_method = condition_method
        self.use_gamma = use_gamma
        self.use_beta = use_beta
        self.use_coords_freq = use_coords
        self.debug_every = debug_every
        self.print_verbose_every = print_verbose_every

        # Initialize helper variables
        self.stem_use_coords = (stem_stride == 1) and (self.use_coords_freq > 0)
        self.condition_pattern = condition_pattern
        if len(condition_pattern) == 0:
            self.condition_pattern = []
            for i in range(self.module_num_layers * self.num_modules):
                self.condition_pattern.append(self.condition_method != 'concat')
        else:
            self.condition_pattern = [i > 0 for i in self.condition_pattern]
        self.extra_channel_freq = self.use_coords_freq
        self.block = FiLMedResBlock
        self.num_cond_maps = 2 * self.module_dim if self.condition_method == 'concat' else 0
        self.fwd_count = 0
        self.num_extra_channels = 2 if self.use_coords_freq > 0 else 0
        if self.debug_every <= -1:
            self.print_verbose_every = 1
        module_H = feature_dim[1] // (stem_stride ** stem_num_layers)  # Rough calc: work for main cases
        module_W = feature_dim[2] // (stem_stride ** stem_num_layers)  # Rough calc: work for main cases
        self.coords = coord_map((module_H, module_W))
        self.default_weight = Variable(torch.ones(1, 1, self.module_dim)).type(torch.cuda.FloatTensor)
        self.default_bias = Variable(torch.zeros(1, 1, self.module_dim)).type(torch.cuda.FloatTensor)

        # Initialize stem
        stem_feature_dim = feature_dim[0] + self.stem_use_coords * self.num_extra_channels
        self.stem = build_stem_film(stem_feature_dim, module_dim,
                                    num_layers=stem_num_layers, with_batchnorm=stem_batchnorm,
                                    kernel_size=stem_kernel_size, stride=stem_stride, padding=stem_padding)

        # Initialize FiLMed network body
        self.function_modules = {}
        self.vocab = vocab
        for fn_num in range(self.num_modules):
            with_cond = self.condition_pattern[self.module_num_layers * fn_num:
                                               self.module_num_layers * (fn_num + 1)]
            mod = self.block(module_dim, with_residual=module_residual, with_batchnorm=module_batchnorm,
                             with_cond=with_cond,
                             dropout=module_dropout,
                             num_extra_channels=self.num_extra_channels,
                             extra_channel_freq=self.extra_channel_freq,
                             with_input_proj=module_input_proj,
                             num_cond_maps=self.num_cond_maps,
                             kernel_size=module_kernel_size,
                             batchnorm_affine=module_batchnorm_affine,
                             num_layers=self.module_num_layers,
                             condition_method=condition_method,
                             debug_every=self.debug_every)
            self.add_module(str(fn_num), mod)
            self.function_modules[fn_num] = mod

        # Initialize output classifier
        self.classifier = build_classifier_film(module_dim + self.num_extra_channels, module_H, module_W,
                                                num_answers, classifier_fc_layers, classifier_proj_dim,
                                                classifier_downsample, with_batchnorm=classifier_batchnorm,
                                                dropout=classifier_dropout)

    def forward(self, x, film, save_activations=False):
        # Initialize forward pass and externally viewable activations
        self.fwd_count += 1
        if save_activations:
            self.feats = None
            self.module_outputs = []
            self.cf_input = None

        # Prepare FiLM layers
        gammas = None
        betas = None
        if self.condition_method == 'concat':
            # Use parameters usually used to condition via FiLM instead to condition via concatenation
            cond_params = film[:, :, :2 * self.module_dim]
            cond_maps = cond_params.unsqueeze(3).unsqueeze(4).expand(cond_params.size() + x.size()[-2:])
        else:
            gammas, betas = torch.split(film[:, :, :2 * self.module_dim], self.module_dim, dim=-1)
            if not self.use_gamma:
                gammas = self.default_weight.expand_as(gammas)
            if not self.use_beta:
                betas = self.default_bias.expand_as(betas)

        # Propagate up image features CNN
        batch_coords = None
        if self.use_coords_freq > 0:
            batch_coords = self.coords.unsqueeze(0).expand(torch.Size((x.size(0), *self.coords.size())))
        if self.stem_use_coords:
            x = torch.cat([x, batch_coords], 1)
        feats = self.stem(x)
        if save_activations:
            self.feats = feats
        N, _, H, W = feats.size()

        # Propagate up the network from low-to-high numbered blocks
        module_inputs = Variable(torch.zeros(feats.size()).unsqueeze(1).expand(
            N, self.num_modules, self.module_dim, H, W)).type(torch.cuda.FloatTensor)
        module_inputs[:, 0] = feats
        for fn_num in range(self.num_modules):
            if self.condition_method == 'concat':
                layer_output = self.function_modules[fn_num](module_inputs[:, fn_num],
                                                             extra_channels=batch_coords,
                                                             cond_maps=cond_maps[:, fn_num])
            else:
                layer_output = self.function_modules[fn_num](module_inputs[:, fn_num],
                                                             gammas[:, fn_num, :], betas[:, fn_num, :], batch_coords)

            # Store for future computation
            if save_activations:
                self.module_outputs.append(layer_output)
            if fn_num == (self.num_modules - 1):
                final_module_output = layer_output
            else:
                module_inputs_updated = module_inputs.clone()
                module_inputs_updated[:, fn_num + 1] = module_inputs_updated[:, fn_num + 1] + layer_output
                module_inputs = module_inputs_updated

        # Run the final classifier over the resultant, post-modulated features.
        if self.use_coords_freq > 0:
            final_module_output = torch.cat([final_module_output, batch_coords], 1)
        if save_activations:
            self.cf_input = final_module_output
        out = self.classifier(final_module_output)
        return out


class FiLMedResBlock(nn.Module):
    def __init__(self, in_dim, out_dim=None, with_residual=True, with_batchnorm=True,
                 with_cond=[False], dropout=0, num_extra_channels=0, extra_channel_freq=1,
                 with_input_proj=0, num_cond_maps=0, kernel_size=3, batchnorm_affine=False,
                 num_layers=1, condition_method='bn-film', debug_every=float('inf')):
        if out_dim is None:
            out_dim = in_dim
        super(FiLMedResBlock, self).__init__()
        self.with_residual = with_residual
        self.with_batchnorm = with_batchnorm
        self.with_cond = with_cond
        self.dropout = dropout
        self.extra_channel_freq = 0 if num_extra_channels == 0 else extra_channel_freq
        self.with_input_proj = with_input_proj  # Kernel size of input projection
        self.num_cond_maps = num_cond_maps
        self.kernel_size = kernel_size
        self.batchnorm_affine = batchnorm_affine
        self.num_layers = num_layers
        self.condition_method = condition_method
        self.debug_every = debug_every

        if self.with_input_proj % 2 == 0:
            raise (NotImplementedError)
        if self.kernel_size % 2 == 0:
            raise (NotImplementedError)
        if self.num_layers >= 2:
            raise (NotImplementedError)

        if self.condition_method == 'block-input-film' and self.with_cond[0]:
            self.film = FiLM()
        if self.with_input_proj:
            self.input_proj = nn.Conv2d(in_dim + (num_extra_channels if self.extra_channel_freq >= 1 else 0),
                                        in_dim, kernel_size=(self.with_input_proj, self.with_input_proj),
                                        padding=(self.with_input_proj // 2, self.with_input_proj // 2))

        self.conv1 = nn.Conv2d(in_dim + self.num_cond_maps +
                               (num_extra_channels if self.extra_channel_freq >= 2 else 0),
                               out_dim, kernel_size=(self.kernel_size, self.kernel_size),
                               padding=(self.kernel_size // 2, self.kernel_size // 2))
        if self.condition_method == 'conv-film' and self.with_cond[0]:
            self.film = FiLM()
        if self.with_batchnorm:
            self.bn1 = nn.BatchNorm2d(out_dim, affine=((not self.with_cond[0]) or self.batchnorm_affine))
        if self.condition_method == 'bn-film' and self.with_cond[0]:
            self.film = FiLM()
        if dropout > 0:
            self.drop = nn.Dropout2d(p=self.dropout)
        if ((self.condition_method == 'relu-film' or self.condition_method == 'block-output-film')
                and self.with_cond[0]):
            self.film = FiLM()

    def forward(self, x, gammas=None, betas=None, extra_channels=None, cond_maps=None):

        if self.condition_method == 'block-input-film' and self.with_cond[0]:
            x = self.film(x, gammas, betas)

        # ResBlock input projection
        if self.with_input_proj:
            if extra_channels is not None and self.extra_channel_freq >= 1:
                x = torch.cat([x, extra_channels], 1)
            x = F.relu(self.input_proj(x))
        out = x

        # ResBlock body
        if cond_maps is not None:
            out = torch.cat([out, cond_maps], 1)
        if extra_channels is not None and self.extra_channel_freq >= 2:
            out = torch.cat([out, extra_channels], 1)
        out = self.conv1(out)
        if self.condition_method == 'conv-film' and self.with_cond[0]:
            out = self.film(out, gammas, betas)
        if self.with_batchnorm:
            out = self.bn1(out)
        if self.condition_method == 'bn-film' and self.with_cond[0]:
            out = self.film(out, gammas, betas)
        if self.dropout > 0:
            out = self.drop(out)
        out = F.relu(out)
        if self.condition_method == 'relu-film' and self.with_cond[0]:
            out = self.film(out, gammas, betas)

        # ResBlock remainder
        if self.with_residual:
            out = x + out
        if self.condition_method == 'block-output-film' and self.with_cond[0]:
            out = self.film(out, gammas, betas)
        return out


def coord_map(shape, start=-1, end=1):
    """
    Gives, a 2d shape tuple, returns two mxn coordinate maps,
    Ranging min-max in the x and y directions, respectively.
    """
    m, n = shape
    x_coord_row = torch.linspace(start, end, steps=n).type(torch.cuda.FloatTensor)
    y_coord_row = torch.linspace(start, end, steps=m).type(torch.cuda.FloatTensor)
    x_coords = x_coord_row.unsqueeze(0).expand(torch.Size((m, n))).unsqueeze(0)
    y_coords = y_coord_row.unsqueeze(1).expand(torch.Size((m, n))).unsqueeze(0)
    return Variable(torch.cat([x_coords, y_coords], 0))


class FiLMGen(nn.Module):
    def __init__(self,
                 null_token=0,
                 start_token=1,
                 end_token=2,
                 encoder_embed=None,
                 encoder_vocab_size=100,
                 decoder_vocab_size=100,
                 wordvec_dim=200,
                 hidden_dim=512,
                 rnn_num_layers=1,
                 rnn_dropout=0,
                 output_batchnorm=False,
                 bidirectional=False,
                 encoder_type='gru',
                 decoder_type='linear',
                 gamma_option='linear',
                 gamma_baseline=1,
                 num_modules=4,
                 module_num_layers=1,
                 module_dim=128,
                 parameter_efficient=False,
                 debug_every=float('inf'),
                 ):
        super(FiLMGen, self).__init__()
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.output_batchnorm = output_batchnorm
        self.bidirectional = bidirectional
        self.num_dir = 2 if self.bidirectional else 1
        self.gamma_option = gamma_option
        self.gamma_baseline = gamma_baseline
        self.num_modules = num_modules
        self.module_num_layers = module_num_layers
        self.module_dim = module_dim
        self.debug_every = debug_every
        self.NULL = null_token
        self.START = start_token
        self.END = end_token
        if self.bidirectional:
            if decoder_type != 'linear':
                raise (NotImplementedError)
            hidden_dim = (int)(hidden_dim / self.num_dir)

        self.func_list = {
            'linear': None,
            'sigmoid': F.sigmoid,
            'tanh': F.tanh,
            'exp': torch.exp,
        }

        self.cond_feat_size = 2 * self.module_dim * self.module_num_layers  # FiLM params per ResBlock
        if not parameter_efficient:  # parameter_efficient=False only used to load older trained models
            self.cond_feat_size = 4 * self.module_dim + 2 * self.num_modules

        self.encoder_embed = nn.Embedding(encoder_vocab_size, wordvec_dim)
        self.encoder_rnn = init_rnn(self.encoder_type, wordvec_dim, hidden_dim, rnn_num_layers,
                                    dropout=rnn_dropout, bidirectional=self.bidirectional)
        self.decoder_rnn = init_rnn(self.decoder_type, hidden_dim, hidden_dim, rnn_num_layers,
                                    dropout=rnn_dropout, bidirectional=self.bidirectional)
        self.decoder_linear = nn.Linear(
            hidden_dim * self.num_dir, self.num_modules * self.cond_feat_size)
        if self.output_batchnorm:
            self.output_bn = nn.BatchNorm1d(self.cond_feat_size, affine=True)

    def get_dims(self, x=None):
        V_in = self.encoder_embed.num_embeddings
        V_out = self.cond_feat_size
        D = self.encoder_embed.embedding_dim
        H = self.encoder_rnn.hidden_size
        H_full = self.encoder_rnn.hidden_size * self.num_dir
        L = self.encoder_rnn.num_layers * self.num_dir

        N = x.size(0) if x is not None else None
        T_in = x.size(1) if x is not None else None
        T_out = self.num_modules
        return V_in, V_out, D, H, H_full, L, N, T_in, T_out

    def before_rnn(self, x, replace=0):
        N, T = x.size()
        idx = torch.LongTensor(N).fill_(T - 1)

        # Find the last non-null element in each sequence.
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
        V_in, V_out, D, H, H_full, L, N, T_in, T_out = self.get_dims(x=x)
        x, idx = self.before_rnn(x)  # Tokenized word sequences (questions), end index
        embed = self.encoder_embed(x)
        h0 = Variable(torch.zeros(L, N, H).type_as(embed.data))

        if self.encoder_type == 'lstm':
            c0 = Variable(torch.zeros(L, N, H).type_as(embed.data))
            out, _ = self.encoder_rnn(embed, (h0, c0))
        elif self.encoder_type == 'gru':
            out, _ = self.encoder_rnn(embed, h0)

        # Pull out the hidden state for the last non-null value in each input
        idx = idx.view(N, 1, 1).expand(N, 1, H_full)
        return out.gather(1, idx).view(N, H_full)

    def decoder(self, encoded, dims, h0=None, c0=None):
        V_in, V_out, D, H, H_full, L, N, T_in, T_out = dims

        if self.decoder_type == 'linear':
            # (N x H) x (H x T_out*V_out) -> (N x T_out*V_out) -> N x T_out x V_out
            return self.decoder_linear(encoded).view(N, T_out, V_out), (None, None)

        encoded_repeat = encoded.view(N, 1, H).expand(N, T_out, H)
        if not h0:
            h0 = Variable(torch.zeros(L, N, H).type_as(encoded.data))

        if self.decoder_type == 'lstm':
            if not c0:
                c0 = Variable(torch.zeros(L, N, H).type_as(encoded.data))
            rnn_output, (ht, ct) = self.decoder_rnn(encoded_repeat, (h0, c0))
        elif self.decoder_type == 'gru':
            ct = None
            rnn_output, ht = self.decoder_rnn(encoded_repeat, h0)

        rnn_output_2d = rnn_output.contiguous().view(N * T_out, H)
        linear_output = self.decoder_linear(rnn_output_2d)
        if self.output_batchnorm:
            linear_output = self.output_bn(linear_output)
        output_shaped = linear_output.view(N, T_out, V_out)
        return output_shaped, (ht, ct)

    def forward(self, x):
        encoded = self.encoder(x)
        film_pre_mod, _ = self.decoder(encoded, self.get_dims(x=x))
        film = self.modify_output(film_pre_mod, gamma_option=self.gamma_option,
                                  gamma_shift=self.gamma_baseline)
        return film

    def modify_output(self, out, gamma_option='linear', gamma_scale=1, gamma_shift=0,
                      beta_option='linear', beta_scale=1, beta_shift=0):
        gamma_func = self.func_list[gamma_option]
        beta_func = self.func_list[beta_option]

        gs = []
        bs = []
        for i in range(self.module_num_layers):
            gs.append(slice(i * (2 * self.module_dim), i * (2 * self.module_dim) + self.module_dim))
            bs.append(slice(i * (2 * self.module_dim) + self.module_dim, (i + 1) * (2 * self.module_dim)))

        if gamma_func is not None:
            for i in range(self.module_num_layers):
                out[:, :, gs[i]] = gamma_func(out[:, :, gs[i]])
        if gamma_scale != 1:
            for i in range(self.module_num_layers):
                out[:, :, gs[i]] = out[:, :, gs[i]] * gamma_scale
        if gamma_shift != 0:
            for i in range(self.module_num_layers):
                out[:, :, gs[i]] = out[:, :, gs[i]] + gamma_shift
        if beta_func is not None:
            for i in range(self.module_num_layers):
                out[:, :, bs[i]] = beta_func(out[:, :, bs[i]])
            # out[:, :, b2] = beta_func(out[:, :, b2])
        if beta_scale != 1:
            for i in range(self.module_num_layers):
                out[:, :, bs[i]] = out[:, :, bs[i]] * beta_scale
        if beta_shift != 0:
            for i in range(self.module_num_layers):
                out[:, :, bs[i]] = out[:, :, bs[i]] + beta_shift
        return out