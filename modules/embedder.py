import torch
import torch.nn as nn
from torch.nn import Module as Module

from bert_modules.bert_modules import BertLayerNorm, PositionalEncoding, BertEncoder
from relation_network_modules.relation_network_modules import RelationalLayer


class DiversityLoss(Module):
    def __init__(self, weight=1.0):
        super(DiversityLoss, self).__init__()
        self.weight = weight

    def forward(self, inputs, targets):
        # Flatten Predictions #
        inputs = torch.argmax(inputs, dim=1).view(-1)

        # Flatten True Labels #
        targets = targets.view(-1)

        # Find all wrong guesses #
        misguesses = (inputs != targets)

        responses = torch.masked_select(inputs, misguesses)

        return self.weight - torch.std(responses.to(float))


class MultiModalEmbedder(Module):
    def __init__(self, config: dict):
        super(MultiModalEmbedder, self).__init__()
        self.config = config
        self.question_embeddings = nn.Embedding(config['question_vocabulary_size'], config['hidden_dim'], padding_idx=0)
        self.position_embeddings = nn.Embedding(
            config['max_objects_per_scene'] + config['max_question_tokens_per_scene'], config['hidden_dim'],
            padding_idx=0)
        self.token_type_embeddings = nn.Embedding(config['num_token_types'], config['hidden_dim'], padding_idx=0)
        self.color_embeddings = nn.Embedding(config['num_colors'] + 1, config['embedding_dim'], padding_idx=0)
        self.shape_embeddings = nn.Embedding(config['num_shapes'] + 1, config['embedding_dim'], padding_idx=0)
        self.material_embeddings = nn.Embedding(config['num_materials'] + 1, config['embedding_dim'], padding_idx=0)
        self.size_embeddings = nn.Embedding(config['num_sizes'] + 1, config['embedding_dim'], padding_idx=0)

        self.position_project = nn.Linear(config['num_positions'], config['embedding_dim'])
        self.reproject = nn.Linear(5 * config['embedding_dim'], config['hidden_dim'])

        self.pros_norm = lambda x: x
        self.scene_norm = lambda x: x
        self.color_norm = lambda x: x
        self.shape_norm = lambda x: x
        self.material_norm = lambda x: x
        self.size_norm = lambda x: x

        self.LayerNormObject = BertLayerNorm(config['hidden_dim'], eps=1e-12)
        self.PosEncQ = PositionalEncoding(config['hidden_dim'], 0.0, 50)
        self.LayerNormQ = BertLayerNorm(config['hidden_dim'], eps=1e-12)
        self.dropout = nn.Dropout(0.1)
        return

    @staticmethod
    def word_to_object_mask(types):
        omask = types.ge(1) * 1.0
        qmask = types.ge(2) * 1.0
        stacked_obj = torch.stack([omask] * omask.size(1), dim=1)
        return torch.einsum("bij,bi->bij", stacked_obj, omask)

    def forward(self,
                positions,
                types,
                object_positions,
                object_colors,
                object_shapes,
                object_materials,
                object_sizes,
                question):
        ### Generate active positions ###
        position_embeddings = self.position_embeddings(positions)

        ### Generate types ####1 CLS -- --4-- Items--- 6 empty -- 50 word question -- #
        type_embeddings = self.token_type_embeddings(types)

        ### Get Tokenized and Padded Questions ###
        ### BS X Q_SEQ_LEN X Reproj Emb
        questions = self.question_embeddings(question)

        ### Generate Attention Mask ###
        mask = types.ge(1) * 1.0
        mask = mask.unsqueeze(1).unsqueeze(2)
        # mask = self.word_to_object_mask(types)
        mask = (1.0 - mask) * -100_000
        # mask = mask.unsqueeze(1)

        ### Generate Object Mask ###
        object_mask = types.eq(1) * 1.0

        ### Gather all positions ###
        ### Projected Version ###
        ### BS X SEQ_LEN X EMB_DIM
        op_proj = self.pros_norm(self.position_project(object_positions))

        ### Gather all colors ###
        ### Get embeddings ###
        oc_proj = self.color_norm(self.color_embeddings(object_colors))

        ### Gather all shapes ###
        ### Get embeddings ###
        os_proj = self.shape_norm(self.shape_embeddings(object_shapes))

        ### Gather all materials ###
        ### Get embeddings ###
        om_proj = self.material_norm(self.material_embeddings(object_materials))

        ### Gather all sizes ###
        ### Get embeddings ###
        oz_proj = self.size_norm(self.size_embeddings(object_sizes))
        object_related_embeddings = torch.cat([op_proj, oc_proj, os_proj, om_proj, oz_proj], 2)
        ### Reproject them to 128 sized-embeddings ###
        ore = self.reproject(object_related_embeddings)

        ore = ore + type_embeddings[:, 0:10, :]
        ore = self.LayerNormObject(ore)

        questions = questions + type_embeddings[:, 10:, :] + position_embeddings[:, 10:, :]
        questions = self.LayerNormQ(questions)

        embeddings = torch.cat([ore, questions], 1)
        return embeddings, mask, object_mask


class SplitModalEmbedder(Module):
    def __init__(self, config: dict):
        super(SplitModalEmbedder, self).__init__()
        self.config = config
        self.question_embeddings = nn.Embedding(config['question_vocabulary_size'], config['embedding_dim'],
                                                padding_idx=0)
        self.color_embeddings = nn.Embedding(config['num_colors'] + 1, config['embedding_dim'], padding_idx=0)
        self.shape_embeddings = nn.Embedding(config['num_shapes'] + 1, config['embedding_dim'], padding_idx=0)
        self.material_embeddings = nn.Embedding(config['num_materials'] + 1, config['embedding_dim'], padding_idx=0)
        self.size_embeddings = nn.Embedding(config['num_sizes'] + 1, config['embedding_dim'], padding_idx=0)
        self.token_type_embeddings = nn.Embedding(config['num_token_types'], config['hidden_dim'], padding_idx=0)
        self.reproject = nn.Linear(3 + 4 * config['embedding_dim'], config['hidden_dim'])
        return

    def forward(self,
                positions,
                types,
                object_positions,
                object_colors,
                object_shapes,
                object_materials,
                object_sizes,
                question):
        type_embeddings = self.token_type_embeddings(types)
        otype_embeddings = type_embeddings[:, 0:10]
        qtype_embeddings = type_embeddings[:, 10:]

        object_mask = types.eq(1) * 1.0
        object_mask = object_mask[:, :10]
        object_mask = object_mask.unsqueeze(1).unsqueeze(2)

        question_mask = types.eq(2) * 1.0
        question_mask = question_mask[:, 10:]
        question_mask = question_mask.unsqueeze(1).unsqueeze(2)

        mixed_mask = torch.cat([object_mask, question_mask], dim=3)

        questions = self.question_embeddings(question)
        questions = questions + qtype_embeddings
        op_proj = object_positions
        oc_proj = self.color_embeddings(object_colors)
        os_proj = self.shape_embeddings(object_shapes)
        om_proj = self.material_embeddings(object_materials)
        oz_proj = self.size_embeddings(object_sizes)
        object_related_embeddings = torch.cat([op_proj, oc_proj, os_proj, om_proj, oz_proj], 2)
        ore = self.reproject(object_related_embeddings)
        ore = ore + otype_embeddings
        return ore, questions, object_mask, question_mask, mixed_mask

class SplitModalEmbedderNoType(Module):
    def __init__(self, config: dict):
        super(SplitModalEmbedderNoType, self).__init__()
        self.config = config
        self.question_embeddings = nn.Embedding(config['question_vocabulary_size'], config['embedding_dim'],
                                                padding_idx=0)
        self.color_embeddings = nn.Embedding(config['num_colors'] + 1, config['embedding_dim'], padding_idx=0)
        self.shape_embeddings = nn.Embedding(config['num_shapes'] + 1, config['embedding_dim'], padding_idx=0)
        self.material_embeddings = nn.Embedding(config['num_materials'] + 1, config['embedding_dim'], padding_idx=0)
        self.size_embeddings = nn.Embedding(config['num_sizes'] + 1, config['embedding_dim'], padding_idx=0)
        self.reproject = nn.Linear(3 + 4 * config['embedding_dim'], config['hidden_dim'])
        return

    def forward(self,
                positions,
                types,
                object_positions,
                object_colors,
                object_shapes,
                object_materials,
                object_sizes,
                question):

        object_mask = types.eq(1) * 1.0
        object_mask = object_mask[:, :10]
        object_mask = object_mask.unsqueeze(1).unsqueeze(2)

        question_mask = types.eq(2) * 1.0
        question_mask = question_mask[:, 10:]
        question_mask = question_mask.unsqueeze(1).unsqueeze(2)

        mixed_mask = torch.cat([object_mask, question_mask], dim=3)

        questions = self.question_embeddings(question)
        questions = questions
        op_proj = object_positions
        oc_proj = self.color_embeddings(object_colors)
        os_proj = self.shape_embeddings(object_shapes)
        om_proj = self.material_embeddings(object_materials)
        oz_proj = self.size_embeddings(object_sizes)
        object_related_embeddings = torch.cat([op_proj, oc_proj, os_proj, om_proj, oz_proj], 2)
        ore = self.reproject(object_related_embeddings)
        return ore, questions, object_mask, question_mask, mixed_mask

class QuestionOnlyEmbedder(Module):
    def __init__(self, config: dict):
        super(QuestionOnlyEmbedder, self).__init__()
        self.config = config
        self.question_embeddings = nn.Embedding(config['question_vocabulary_size'], config['embedding_dim'],
                                                padding_idx=0)
        return

    def forward(self,
                positions,
                types,
                object_positions,
                object_colors,
                object_shapes,
                object_materials,
                object_sizes,
                question):
        question_mask = types.eq(2) * 1.0
        question_mask = question_mask[:, 10:]
        question_mask = question_mask.unsqueeze(1).unsqueeze(2)
        questions = self.question_embeddings(question)
        return questions, question_mask


class MLPClassifierHead(Module):
    def __init__(self, config: dict, use_log_transform=False, mode='raw'):
        super(MLPClassifierHead, self).__init__()
        self.linear_layer_1 = nn.Linear(config['hidden_dim'], config['hidden_dim'])
        self.linear_layer_2 = nn.Linear(config['hidden_dim'], config['num_output_classes'])

        if mode == 'arg':
            self.softmax_layer = lambda x: torch.argmax(x, dim=1, keepdim=True)
        elif mode == 'soft':
            if use_log_transform:
                self.softmax_layer = nn.LogSoftmax(dim=1)
            else:
                self.softmax_layer = nn.Softmax(dim=1)
        elif mode == 'raw':
            self.softmax_layer = lambda x: x
        else:
            raise NotImplementedError(f"Mode: {mode} not implemented in MLPClassifierHead Module...")
        return

    def forward(self, input):
        input = nn.ReLU()(self.linear_layer_1(input))
        return self.softmax_layer(self.linear_layer_2(input))


class ConcatClassifierHead(Module):
    def __init__(self, config: dict):
        super(ConcatClassifierHead, self).__init__()
        self.linear_layer_1 = nn.Linear(config['max_objects_per_scene'] * config['hidden_dim'], config['hidden_dim'])
        self.linear_layer_2 = nn.Linear(config['hidden_dim'], config['num_output_classes'])

    def forward(self, input_set):
        flat_set = input_set.view(input_set.size(0), -1)
        flat_set = nn.ReLU()(self.linear_layer_1(flat_set))
        return self.linear_layer_2(flat_set)


class PerOutputClassifierHead(Module):
    def __init__(self, config: dict):
        super(PerOutputClassifierHead, self).__init__()
        self.linear_layer_1 = nn.Linear(config['hidden_dim'], config['hidden_dim'] // 2)
        self.linear_layer_2 = nn.Linear(config['hidden_dim'] // 2, config['num_output_classes'])

    def forward(self, input_set):
        reduced_set = torch.sum(input_set, dim=1)
        reduced_set = self.linear_layer_1(reduced_set)
        reduced_set = nn.ReLU()(reduced_set)
        reduced_set = self.linear_layer_2(reduced_set)
        return reduced_set


class QuestionEmbedModel(Module):
    def __init__(self, config: dict):
        super(QuestionEmbedModel, self).__init__()
        self.bidirectional = bool(config['use_bidirectional_encoder'])
        self.lstm = nn.LSTM(config['embedding_dim'], config['hidden_dim'], batch_first=True,
                            bidirectional=self.bidirectional)
        self.reduce = nn.Linear(2 * config['hidden_dim'], config['hidden_dim'])

    def forward(self, question):
        self.lstm.flatten_parameters()
        _, (h, c) = self.lstm(question)
        h = torch.transpose(h, 1, 0)
        h = h.reshape(h.size(0), h.size(1) * h.size(2))
        h = self.reduce(h)
        return h


class DeltaFormer(Module):
    def __init__(self, config: dict):
        super(DeltaFormer, self).__init__()
        self.mme = MultiModalEmbedder(config)
        self.be = BertEncoder(config)
        self.classhead = MLPClassifierHead(config)
        self.concathead = ConcatClassifierHead(config)
        self.perhead = PerOutputClassifierHead(config)
        return

    def forward(self, **kwargs):
        embeddings, mask, obj_mask = self.mme(**kwargs)
        out, atts = self.be.forward(embeddings, mask, output_all_encoded_layers=False, output_attention_probs=True)
        item_output = out[-1][:, 0:10]
        filtered_item_output = item_output * obj_mask[:, 0:10].unsqueeze(2)
        answer = self.perhead(filtered_item_output)
        return answer, atts[-1], None


class DeltaRN(Module):
    def __init__(self, config: dict):
        super(DeltaRN, self).__init__()
        self.sme = SplitModalEmbedderNoType(config)
        self.seq = QuestionEmbedModel(config)
        self.rn = RelationalLayer(config)
        return

    def forward(self, **kwargs):
        ore, questions, object_mask, question_mask, _ = self.sme(**kwargs)
        qst = self.seq(questions)
        answer = self.rn(ore, qst)

        return answer, None, None


class DeltaSQFormer(Module):
    def __init__(self, config: dict):
        super(DeltaSQFormer, self).__init__()
        self.sme = SplitModalEmbedder(config)
        self.pos_enc = PositionalEncoding(d_model=config['embedding_dim'], dropout=0.1, max_len=50)
        self.oln = BertLayerNorm(config['hidden_dim'])
        self.qln = BertLayerNorm(config['hidden_dim'])
        self.be = BertEncoder(config)
        self.classhead = MLPClassifierHead(config)
        self.avghead = PerOutputClassifierHead(config)
        return

    def forward(self, **kwargs):
        object_emb, question_emb, _, _, mixed_mask = self.sme(**kwargs)
        object_emb = self.oln(object_emb)
        question_emb = self.pos_enc(question_emb)
        question_emb = self.qln(question_emb)
        embeddings = torch.cat([object_emb, question_emb], dim=1)
        out, atts = self.be.forward(embeddings, mixed_mask, output_all_encoded_layers=False,
                                    output_attention_probs=True)
        item_output = out[-1][:, 0]
        answer = self.classhead(item_output)

        # answer = self.avghead(out[-1])
        return answer, atts[-1], None


class DeltaQFormer(Module):
    def __init__(self, config: dict):
        super(DeltaQFormer, self).__init__()
        self.qe = QuestionOnlyEmbedder(config)
        self.pos_enc = PositionalEncoding(d_model=config['embedding_dim'], dropout=0.1, max_len=50)
        self.be = BertEncoder(config)
        self.classhead = MLPClassifierHead(config)
        return

    def forward(self, **kwargs):
        embeddings, mask = self.qe(**kwargs)
        embeddings = self.pos_enc(embeddings)
        out, atts = self.be.forward(embeddings, mask, output_all_encoded_layers=False, output_attention_probs=True)
        item_output = out[-1][:, 0]
        answer = self.classhead(item_output)
        return answer, atts[-1], None
