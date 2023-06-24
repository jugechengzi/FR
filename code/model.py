import torch.nn as nn
import torch
import torch.nn.functional as F


class Rnn(nn.Module):

    def __init__(self, cell_type, embedding_dim, hidden_dim, num_layers):
        super(Rnn, self).__init__()
        if cell_type == 'LSTM':
            self.rnn = nn.LSTM(input_size=embedding_dim,
                               hidden_size=hidden_dim // 2,
                               num_layers=num_layers,
                               batch_first=True,
                               bidirectional=True)
        elif cell_type == 'GRU':
            self.rnn = nn.GRU(input_size=embedding_dim,
                              hidden_size=hidden_dim // 2,
                              num_layers=num_layers,
                              batch_first=True,
                              bidirectional=True)
        else:
            raise NotImplementedError('cell_type {} is not implemented'.format(cell_type))

    def forward(self, x):
        """
        Inputs:
        x - - (batch_size, seq_length, input_dim)
        Outputs:
        h - - bidirectional(batch_size, seq_length, hidden_dim)
        """
        h = self.rnn(x)
        return h


class SelectItem(nn.Module):
    def __init__(self, item_index):
        super(SelectItem, self).__init__()
        self._name = 'selectitem'
        self.item_index = item_index

    def forward(self, inputs):
        return inputs[self.item_index]

class Embedding(nn.Module):

    def __init__(self, vocab_size, embedding_dim, pretrained_embedding=None):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if pretrained_embedding is not None:
            self.embedding.weight.data = torch.from_numpy(pretrained_embedding)
        self.embedding.weight.requires_grad = False

    def forward(self, x):
        """
        Inputs:
        x -- (batch_size, seq_length)
        Outputs
        shape -- (batch_size, seq_length, embedding_dim)
        """
        return self.embedding(x)


class Attention(nn.Module):

    def __init__(self, args):
        super(Attention, self).__init__()

        self.proj = nn.Linear(args.hidden_dim, args.hidden_dim)

        self.head = nn.Linear(args.hidden_dim, 1)

    def forward(self, inputs, masks):
        """
        Inputs:
            inputs -- (batch_size, seq_length, input_dim)
            masks -- (batch_size, seq_length, 1)
        Outputs:
            output -- (batch_size, input_dim)
            atts -- (batch_size, seq_length)
        """
        proj_inputs = self.proj(inputs)  # (batch_size, seq_length, hidden_dim)
        att_logits = self.head(proj_inputs)  # (batch_size, seq_length, 1)

        # <pad> should have attention score 0
        neg_inf = -1e9
        att_logits = att_logits * masks + (1. - masks) * neg_inf

        atts = torch.softmax(att_logits, axis=1)  # (batch_size, seq_length, 1)
        atts = atts * masks

        context_vecs = inputs * atts  # (batch_size, seq_length, input_dim)
        output = torch.sum(context_vecs, axis=1)

        return output, torch.squeeze(atts, axis=-1)


class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.args = args

        # initialize embedding layers
        self.embedding_layer = Embedding(args.vocab_size,
                                         args.embedding_dim,
                                         args.pretrained_embedding)
        # initialize a RNN encoder
        self.encoder = Rnn(args.cell_type, args.embedding_dim, args.hidden_dim, args.num_layers)
        # initialize a fc layer
        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(args.hidden_dim, args.num_class)

    def forward(self, inputs, masks, z=None):
        """
        Inputs:
            inputs -- (batch_size, seq_length)
            masks -- (batch_size, seq_length)
        Outputs:
            logits -- (batch_size, num_class)
        """
        # expand dim for masks
        masks_ = masks.unsqueeze(-1)
        # (batch_siz, seq_length, embedding_dim)
        embeddings = masks_ * self.embedding_layer(inputs)
        # (batch_siz, seq_length, embedding_dim * 2)
        if z is not None:
            embeddings = embeddings * (z.unsqueeze(-1))
        outputs, _ = self.encoder(embeddings)

        # mask before max pooling
        outputs = outputs * masks_ + (1. - masks_) * (-1e6)

        # (batch_size, hidden_dim * 2, seq_length)
        outputs = torch.transpose(outputs, 1, 2)
        outputs, _ = torch.max(outputs, axis=2)
        # shape -- (batch_size, num_classes)
        logits = self.fc(self.dropout(outputs))
        return logits


class TargetRnn(nn.Module):
    def __init__(self, args):
        super(TargetRnn, self).__init__()
        self.args = args
        self.gen_embedding_layer = Embedding(args.vocab_size,
                                             args.embedding_dim,
                                             args.pretrained_embedding)
        self.cls_embedding_layer = Embedding(args.vocab_size,
                                             args.embedding_dim,
                                             args.pretrained_embedding)
        self.gen_encoder = Rnn(args.cell_type, args.embedding_dim, args.hidden_dim, args.num_layers)
        self.cls_encoder = Rnn(args.cell_type, args.embedding_dim, args.hidden_dim, args.num_layers)
        self.att_layer = Attention(args)
        self.cls_fc = nn.Linear(args.hidden_dim, args.num_class)
        self.dropout = nn.Dropout(args.dropout)

    def _independent_soft_sampling(self, rationale_logits):
        """
        Use the hidden states at all time to sample whether each word is a rationale or not.
        No dependency between actions. Return the sampled (soft) rationale mask.
        Outputs:
                z -- (batch_size, sequence_length, 2)
        """
        z = torch.softmax(rationale_logits, dim=-1)

        return z

    def independent_straight_through_sampling(self, rationale_logits):
        """
        Straight through sampling.
        Outputs:
            z -- shape (batch_size, sequence_length, 2)
        """
        z = self._independent_soft_sampling(rationale_logits)
        z = F.gumbel_softmax(rationale_logits, tau=1, hard=True)
        return z

    def forward(self, inputs, masks):
        masks_ = masks.unsqueeze(-1)

        ###### Generator ######
        gen_embedding = masks_ * self.gen_embedding_layer(inputs)
        gen_inputs, _ = self.gen_encoder(gen_embedding)
        _, att_score = self.att_layer(gen_inputs, masks_)

        # choose rationale from att_score
        # sample rationale (batch_size, seq_length)
        rationale = self.independent_straight_through_sampling(att_score)

        ###### Classifier ######
        cls_embedding = masks_ * self.cls_embedding_layer(inputs)
        cls_inputs = cls_embedding * rationale.unsqueeze(-1)
        cls_outputs, _ = self.cls_encoder(cls_inputs)
        cls_outputs = cls_outputs * masks_ + (1. -
                                              masks_) * (-1e6)
        # (batch_size, hidden_dim, seq_length)
        cls_outputs = torch.transpose(cls_outputs, 1, 2)
        cls_outputs = torch.mean(cls_outputs, axis=2)
        # shape -- (batch_size, num_classes)
        cls_logits = self.cls_fc(self.dropout(cls_outputs))

        return att_score, rationale, cls_logits


class AttentionGenerator(nn.Module):

    def __init__(self, args):
        super(AttentionGenerator, self).__init__()
        self.args = args
        self.embedding_layer = Embedding(args.vocab_size,
                                         args.embedding_dim,
                                         args.pretrained_embedding)
        self.gen_layer = Rnn(args.cell_type,
                             args.embedding_dim,
                             args.hidden_dim,
                             args.num_layers)
        self.attention_layer = nn.MultiheadAttention(args.embedding_dim, 2, batch_first=True)
        self.z_dim = 2
        self.fc_layer = nn.Linear(args.hidden_dim, self.z_dim)

    def _independent_soft_sampling(self, rationale_logits):
        """
        Use the hidden states at all time to sample whether each word is a rationale or not.
        No dependency between actions. Return the sampled (soft) rationale mask.
        Outputs:
                z -- (batch_size, sequence_length, 2)
        """
        z = torch.softmax(rationale_logits, dim=-1)

        return z

    def independent_straight_through_sampling(self, rationale_logits):
        """
        Straight through sampling.
        Outputs:
            z -- shape (batch_size, sequence_length, 2)
        """
        z = self._independent_soft_sampling(rationale_logits)
        z = F.gumbel_softmax(rationale_logits, tau=1, hard=True)
        return z

    def forward(self, inputs, masks):
        masks_ = masks.unsqueeze(-1)
        ############## generator ##############
        embedding = masks_ * self.embedding_layer(inputs)
        gen_outputs, _ = self.gen_layer(embedding)

        ############## attentions ##############
        att_output, _ = self.attention_layer(gen_outputs, gen_outputs, gen_outputs)

        rationale_logit = self.fc_layer(att_output)

        ############## rationales ##############
        rationales = self.independent_straight_through_sampling(rationale_logit)

        return rationales


class GenEncShareModel(nn.Module):

    def __init__(self, args):
        super(GenEncShareModel, self).__init__()
        self.args = args
        self.embedding_layer = Embedding(args.vocab_size,
                                         args.embedding_dim,
                                         args.pretrained_embedding)
        self.enc = Rnn(args.cell_type,
                       args.embedding_dim,
                       args.hidden_dim,
                       args.num_layers)
        self.z_dim = 2
        self.gen_fc = nn.Linear(args.hidden_dim, self.z_dim)
        self.cls_fc = nn.Linear(args.hidden_dim, args.num_class)
        self.dropout = nn.Dropout(args.dropout)
        self.layernorm = nn.LayerNorm(args.hidden_dim)

    def _independent_soft_sampling(self, rationale_logits):
        """
        Use the hidden states at all time to sample whether each word is a rationale or not.
        No dependency between actions. Return the sampled (soft) rationale mask.
        Outputs:
                z -- (batch_size, sequence_length, 2)
        """
        z = torch.softmax(rationale_logits, dim=-1)

        return z

    def independent_straight_through_sampling(self, rationale_logits):
        """
        Straight through sampling.
        Outputs:
            z -- shape (batch_size, sequence_length, 2)
        """
        z = self._independent_soft_sampling(rationale_logits)
        z = F.gumbel_softmax(rationale_logits, tau=self.args.tau[0], hard=True)
        return z

    # inputs (batch_size, seq_length)
    # masks (batch_size, seq_length)
    def forward(self, inputs, masks):
        #  masks_ (batch_size, seq_length, 1)
        masks_ = masks.unsqueeze(-1)
        ########## Genetator ##########
        embedding = masks_ * self.embedding_layer(inputs)  # (batch_size, seq_length, embedding_dim)
        gen_output, _ = self.enc(embedding)  # (batch_size, seq_length, hidden_dim)
        gen_output = self.layernorm(gen_output)
        gen_logits = self.gen_fc(self.dropout(gen_output))  # (batch_size, seq_length, 2)
        ########## Sample ##########
        z = self.independent_straight_through_sampling(gen_logits)  # (batch_size, seq_length, 2)
        ########## Classifier ##########
        cls_embedding = embedding * (z[:, :, 1].unsqueeze(-1))  # (batch_size, seq_length, embedding_dim)
        cls_outputs, _ = self.enc(cls_embedding)  # (batch_size, seq_length, hidden_dim)
        cls_outputs = self.layernorm(cls_outputs)
        cls_outputs = cls_outputs * masks_ + (1. -
                                              masks_) * (-1e6)
        # (batch_size, hidden_dim, seq_length)
        cls_outputs = torch.transpose(cls_outputs, 1, 2)
        cls_outputs, _ = torch.max(cls_outputs, axis=2)
        # shape -- (batch_size, num_classes)
        cls_logits = self.cls_fc(self.dropout(cls_outputs))

        # LSTM
        return z, cls_logits

    def train_one_step(self, inputs, masks):
        masks_ = masks.unsqueeze(-1)
        # (batch_size, seq_length, embedding_dim)
        embedding = masks_ * self.embedding_layer(inputs)
        outputs, _ = self.enc(embedding)  # (batch_size, seq_length, hidden_dim)
        outputs = self.layernorm(outputs)
        outputs = outputs * masks_ + (1. -
                                      masks_) * (-1e6)
        # (batch_size, hidden_dim, seq_length)
        outputs = torch.transpose(outputs, 1, 2)
        outputs, _ = torch.max(outputs, axis=2)
        # shape -- (batch_size, num_classes)
        logits = self.cls_fc(self.dropout(outputs))
        return logits




    def g_skew(self,inputs, masks):
        #  masks_ (batch_size, seq_length, 1)
        masks_ = masks.unsqueeze(-1)
        ########## Genetator ##########
        embedding = masks_ * self.embedding_layer(inputs)  # (batch_size, seq_length, embedding_dim)
        gen_output, _ = self.enc(embedding)  # (batch_size, seq_length, hidden_dim)
        gen_output = self.layernorm(gen_output)
        gen_logits = self.gen_fc(self.dropout(gen_output))  # (batch_size, seq_length, 2)
        soft_log=self._independent_soft_sampling(gen_logits)
        return soft_log






