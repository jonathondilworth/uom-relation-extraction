"""
GCN model for relation extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from model.tree import Tree, head_to_tree, tree_to_adj
from utils import constant, torch_utils

# --------------------------- ADDED CODE ----------------------------
from torchtune.modules import RotaryPositionalEmbeddings
# -------------------------------------------------------------------


class GCNClassifier(nn.Module):
    """ A wrapper classifier for GCNRelationModel. """
    def __init__(self, opt, emb_matrix=None):
        super().__init__()
        self.gcn_model = GCNRelationModel(opt, emb_matrix=emb_matrix)
        in_dim = opt['hidden_dim']
        self.classifier = nn.Linear(in_dim, opt['num_class'])
        self.opt = opt

    def conv_l2(self):
        return self.gcn_model.gcn.conv_l2()

    def forward(self, inputs):
        outputs, pooling_output = self.gcn_model(inputs)
        logits = self.classifier(outputs)
        return logits, pooling_output

class GCNRelationModel(nn.Module):
    def __init__(self, opt, emb_matrix=None):
        super().__init__()
        self.opt = opt
        self.emb_matrix = emb_matrix

        # create embedding layers
        self.emb = nn.Embedding(opt['vocab_size'], opt['emb_dim'], padding_idx=constant.PAD_ID)
        self.pos_emb = nn.Embedding(len(constant.POS_TO_ID), opt['pos_dim']) if opt['pos_dim'] > 0 else None
        self.ner_emb = nn.Embedding(len(constant.NER_TO_ID), opt['ner_dim']) if opt['ner_dim'] > 0 else None
        embeddings = (self.emb, self.pos_emb, self.ner_emb)
        self.init_embeddings()

        # gcn layer
        self.gcn = GCN(opt, embeddings, opt['hidden_dim'], opt['num_layers'])

        # output mlp layers
        # ------------------------- ADAPTATIONS ---------------------------
        # a sentence is embedded by a vector of length opt['hidden_dim']
        if opt['use_sentence_emb']:
            in_dim = opt['hidden_dim']*3
        else:
            in_dim = opt['hidden_dim']*2
        # -----------------------------------------------------------------
        
        layers = [nn.Linear(in_dim, opt['hidden_dim']), nn.ReLU()]
        for _ in range(self.opt['mlp_layers']-1):
            layers += [nn.Linear(opt['hidden_dim'], opt['hidden_dim']), nn.ReLU()]
        self.out_mlp = nn.Sequential(*layers)
        
        # --------------------------- ADDED CODE ----------------------------
        # create an attention layer
        if opt['attention']:
            # rotary positional encoding
            if opt['positional_emb'] == 'rot':
                # embedding per head
                self.posit_emb = RotaryPositionalEmbeddings(opt['hidden_dim'] // opt['num_heads'])
                
            if opt['pool_before_attention']:
                # create separate attention blocks for subject, object and optionally sentence
                self.mh_attention_subj = torch.nn.MultiheadAttention(embed_dim=opt['hidden_dim'],
                                                                     num_heads=opt['num_heads'],
                                                                     dropout=opt['attention_dropout'],
                                                                     batch_first=True)
                self.mh_attention_obj = torch.nn.MultiheadAttention(embed_dim=opt['hidden_dim'],
                                                                    num_heads=opt['num_heads'],
                                                                    dropout=opt['attention_dropout'],
                                                                    batch_first=True)
                if opt['use_sentence_emb']:
                    self.mh_attention_sent = torch.nn.MultiheadAttention(embed_dim=opt['hidden_dim'],
                                                                         num_heads=opt['num_heads'],
                                                                         dropout=opt['attention_dropout'],
                                                                         batch_first=True)
            else:
                self.mh_attention = torch.nn.MultiheadAttention(embed_dim=opt['hidden_dim'],
                                                                num_heads=opt['num_heads'],
                                                                dropout=opt['attention_dropout'],
                                                                batch_first=True)
        # -------------------------------------------------------------------

    def init_embeddings(self):
        if self.emb_matrix is None:
            self.emb.weight.data[1:,:].uniform_(-1.0, 1.0)
        else:
            self.emb_matrix = torch.from_numpy(self.emb_matrix)
            self.emb.weight.data.copy_(self.emb_matrix)
        # decide finetuning
        if self.opt['topn'] <= 0:
            print("Do not finetune word embedding layer.")
            self.emb.weight.requires_grad = False
        elif self.opt['topn'] < self.opt['vocab_size']:
            print("Finetune top {} word embeddings.".format(self.opt['topn']))
            self.emb.weight.register_hook(lambda x: \
                    torch_utils.keep_partial_grad(x, self.opt['topn']))
        else:
            print("Finetune all embeddings.")

    def forward(self, inputs):
        words, masks, pos, ner, deprel, head, subj_pos, obj_pos, subj_type, obj_type = inputs # unpack
        l = (masks.data.cpu().numpy() == 0).astype(np.int64).sum(1)
        maxlen = max(l)

        def inputs_to_tree_reps(head, words, l, prune, subj_pos, obj_pos):
            head, words, subj_pos, obj_pos = head.cpu().numpy(), words.cpu().numpy(), subj_pos.cpu().numpy(), obj_pos.cpu().numpy()
            trees = [head_to_tree(head[i], words[i], l[i], prune, subj_pos[i], obj_pos[i]) for i in range(len(l))]
            adj = [tree_to_adj(maxlen, tree, directed=False, self_loop=False).reshape(1, maxlen, maxlen) for tree in trees]
            adj = np.concatenate(adj, axis=0)
            adj = torch.from_numpy(adj)
            return Variable(adj.cuda()) if self.opt['cuda'] else Variable(adj)

        adj = inputs_to_tree_reps(head.data, words.data, l, self.opt['prune_k'], subj_pos.data, obj_pos.data)
        h, pool_mask = self.gcn(adj, inputs)
        
        # masks
        subj_mask, obj_mask = subj_pos.eq(0).eq(0).unsqueeze(2), obj_pos.eq(0).eq(0).unsqueeze(2) # invert mask
        pool_type = self.opt['pooling']
        
        # ------------------------ ADAPTATIONS ---------------------------
        if self.opt['positional_emb'] == 'rot':
            # [b, m, h] -> [b, m, heads, h per head]
            h = h.unflatten(-1, (self.opt['num_heads'], self.opt['hidden_dim'] // self.opt['num_heads']))
            # apply positional embeddings
            h = self.posit_emb(h)
            # collapse heads
            h = h.flatten(-2)
        
        h_out = None
        if self.opt['pool_before_attention'] or not self.opt['attention']:
            if self.opt['use_sentence_emb']:
                h_out = pool(h, pool_mask, type=pool_type)
            subj_out = pool(h, subj_mask, type=pool_type)
            obj_out = pool(h, obj_mask, type=pool_type)
        # -----------------------------------------------------------------
        
        # --------------------------- MY CODE ----------------------------
        if self.opt['attention']:
            if self.opt['pool_before_attention']:
                # cross-attention
                # ([b, h], [b, h]) -> ([b, 1, h], [b, 1, h]) -> [b, 2, h]
                queries = torch.cat((subj_out.unsqueeze(1), obj_out.unsqueeze(1)), dim=1)
                if self.opt['use_sentence_emb']:
                    # ([b, 2, h], [b, 1, h]) -> [b, 3, h]
                    queries = torch.cat((queries, h_out.unsqueeze(1)), dim=1)
                
                # [b, h] -> [b, 1, h]
                query_subj = subj_out.unsqueeze(1)
                query_obj = obj_out.unsqueeze(1)
                # get attention values for subject and object
                attn_subj, _ = self.mh_attention_subj(query=query_subj, key=h, value=h, key_padding_mask=pool_mask.squeeze(-1))
                attn_obj, _ = self.mh_attention_obj(query=query_obj, key=h, value=h, key_padding_mask=pool_mask.squeeze(-1))
                # residual connection
                attn_subj = query_subj + attn_subj
                attn_obj = query_obj + attn_obj
                
                # [b, 1, h]; [b, 1, h] -> [b, 2, h]
                mlp_inputs = torch.cat((attn_subj, attn_obj), dim=1)
                
                # sentence embeddings
                if self.opt['use_sentence_emb']:
                    # [b, h] -> [b, 1, h]
                    query_sent = h_out.unsqueeze(1)
                    # get attention values for sentence
                    attn_sent, _ = self.mh_attention_sent(query=query_sent, key=h, value=h, key_padding_mask=pool_mask.squeeze(-1))
                    # residual
                    attn_sent = query_sent + attn_sent
                    # [b, 2, h]; [b, 1, h] -> [b, 3, h]
                    mlp_inputs = torch.cat((mlp_inputs, attn_sent), dim=1)
                
                # [b, c, h] -> [b, c*h]
                mlp_inputs = mlp_inputs.flatten(-2)
            else:
                # self-attention
                attn_out, _ = self.mh_attention(query=h, key=h, value=h, key_padding_mask=pool_mask.squeeze(-1))
                # residual
                attn_out = h + attn_out
                # pool after attention
                subj_out = pool(attn_out, subj_mask, type=pool_type)
                obj_out = pool(attn_out, obj_mask, type=pool_type)
                mlp_inputs = torch.cat((subj_out, obj_out), dim=1)
                if self.opt['use_sentence_emb']:
                    h_out = pool(attn_out, pool_mask, type=pool_type)
                    mlp_inputs = torch.cat((mlp_inputs, h_out), dim=1)
        else:
            mlp_inputs = torch.cat([subj_out, obj_out], dim=1)
            if self.opt['use_sentence_emb']:
                mlp_inputs = torch.cat((mlp_inputs, h_out), dim=1)
        
        # run classification mlp
        outputs = self.out_mlp(mlp_inputs)
        # initialize h_out as the vector of zeros to conform with the existing architecture in the case
        # that we do not use sentence embeddings
        if h_out is None:
            h_out = torch.zeros((h.shape[0], self.opt['hidden_dim']))
        # -----------------------------------------------------------------
        
        return outputs, h_out

class GCN(nn.Module):
    """ A GCN/Contextualized GCN module operated on dependency graphs. """
    def __init__(self, opt, embeddings, mem_dim, num_layers):
        super(GCN, self).__init__()
        self.opt = opt
        self.layers = num_layers
        self.use_cuda = opt['cuda']
        self.mem_dim = mem_dim
        self.in_dim = opt['emb_dim'] + opt['pos_dim'] + opt['ner_dim']

        self.emb, self.pos_emb, self.ner_emb = embeddings

        # rnn layer
        if self.opt.get('rnn', False):
            input_size = self.in_dim
            self.rnn = nn.LSTM(input_size, opt['rnn_hidden'], opt['rnn_layers'], batch_first=True, \
                    dropout=opt['rnn_dropout'], bidirectional=True)
            self.in_dim = opt['rnn_hidden'] * 2
            self.rnn_drop = nn.Dropout(opt['rnn_dropout']) # use on last layer output

        self.in_drop = nn.Dropout(opt['input_dropout'])
        self.gcn_drop = nn.Dropout(opt['gcn_dropout'])

        # gcn layer
        self.W = nn.ModuleList()

        for layer in range(self.layers):
            input_dim = self.in_dim if layer == 0 else self.mem_dim
            self.W.append(nn.Linear(input_dim, self.mem_dim))

    def conv_l2(self):
        conv_weights = []
        for w in self.W:
            conv_weights += [w.weight, w.bias]
        return sum([x.pow(2).sum() for x in conv_weights])

    def encode_with_rnn(self, rnn_inputs, masks, batch_size):
        seq_lens = list(masks.data.eq(constant.PAD_ID).long().sum(1).squeeze())
        # ------------------------ ADAPTATIONS ----------------------------
        # allow the pipeline to run without cuda
        h0, c0 = rnn_zero_state(batch_size, self.opt['rnn_hidden'], self.opt['rnn_layers'], use_cuda=self.opt.get('cuda', False))
        # ------------------------------------------------------------------
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, seq_lens, batch_first=True)
        rnn_outputs, (ht, ct) = self.rnn(rnn_inputs, (h0, c0))
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
        return rnn_outputs

    def forward(self, adj, inputs):
        words, masks, pos, ner, deprel, head, subj_pos, obj_pos, subj_type, obj_type = inputs # unpack
        word_embs = self.emb(words)
        embs = [word_embs]
        if self.opt['pos_dim'] > 0:
            embs += [self.pos_emb(pos)]
        if self.opt['ner_dim'] > 0:
            embs += [self.ner_emb(ner)]
        embs = torch.cat(embs, dim=2)
        embs = self.in_drop(embs)

        # rnn layer
        if self.opt.get('rnn', False):
            gcn_inputs = self.rnn_drop(self.encode_with_rnn(embs, masks, words.size()[0]))
        else:
            gcn_inputs = embs
        
        # gcn layer
        denom = adj.sum(2).unsqueeze(2) + 1
        mask = (adj.sum(2) + adj.sum(1)).eq(0).unsqueeze(2)
        # zero out adj for ablation
        if self.opt.get('no_adj', False):
            adj = torch.zeros_like(adj)

        for l in range(self.layers):
            Ax = adj.bmm(gcn_inputs)
            AxW = self.W[l](Ax)
            AxW = AxW + self.W[l](gcn_inputs) # self loop
            AxW = AxW / denom

            gAxW = F.relu(AxW)
            gcn_inputs = self.gcn_drop(gAxW) if l < self.layers - 1 else gAxW

        return gcn_inputs, mask

def pool(h, mask, type='max'):
    if type == 'max':
        h = h.masked_fill(mask, -constant.INFINITY_NUMBER)
        return torch.max(h, 1)[0]
    elif type == 'avg':
        h = h.masked_fill(mask, 0)
        return h.sum(1) / (mask.size(1) - mask.float().sum(1))
    else:
        h = h.masked_fill(mask, 0)
        return h.sum(1)

def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True, use_cuda=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    if use_cuda:
        return h0.cuda(), c0.cuda()
    else:
        return h0, c0

