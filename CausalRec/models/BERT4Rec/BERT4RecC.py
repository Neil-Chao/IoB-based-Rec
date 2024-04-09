import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))
import torch
import torch.nn as nn

from CausalRec.models.BERT4Rec.bert_modules.bert import BERT
from CausalRec.models.BERT4Rec.bert_modules.utils.outlayer import OutLayer
from CausalRec.models.BERT4Rec.bert_modules.utils.causal_layer import CausalLayer


import argparse


class BERT4RecC(nn.Module):
    '''
    bert_max_len: Length of sequence for bert
    num_items: Number of total items
    bert_num_blocks: Number of transformer layers
    bert_num_heads: Number of heads for multi-attention
    bert_hidden_units: Size of hidden vectors (d_model)
    bert_dropout: Dropout probability to use throughout the model
    '''
    def __init__(self, bert_max_len, num_items, ae, w, bert_num_blocks=2, bert_num_heads=4, bert_hidden_units=32, bert_dropout=0.5, bert_mask_prob=0.2, t=0.3, gamma1=5, gamma2=5) -> None:
        super().__init__()
        
        parser = argparse.ArgumentParser(description='RecPlay')
        parser.add_argument('--bert_max_len', type=int, default=None, help='Length of sequence for bert')
        parser.add_argument('--num_items', type=int, default=None, help='Number of total items')
        parser.add_argument('--bert_hidden_units', type=int, default=None, help='Size of hidden vectors (d_model)')
        parser.add_argument('--bert_num_blocks', type=int, default=None, help='Number of transformer layers')
        parser.add_argument('--bert_num_heads', type=int, default=None, help='Number of heads for multi-attention')
        parser.add_argument('--bert_dropout', type=float, default=None, help='Dropout probability to use throughout the model')
        parser.add_argument('--bert_mask_prob', type=float, default=None, help='Probability for masking items in the training sequence')
        parser.add_argument('d', default=None, help='dddd')
        args = parser.parse_args()
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.ae = self._handle_ae(ae)
        self.w = self._filter_w(w, t)
        args.bert_dropout = bert_dropout
        args.bert_hidden_units = bert_hidden_units
        args.bert_mask_prob = bert_mask_prob
        args.bert_max_len = bert_max_len
        args.bert_num_blocks = bert_num_blocks
        args.bert_num_heads = bert_num_heads
        args.num_items = num_items
        args.model_init_seed = 0


        self.bert = BERT(args)
        # self.out = nn.Linear(self.bert.hidden, num_items + 1)
        self.out = OutLayer(self.bert.hidden, num_items)


        self.cl = CausalLayer(self.ae, self.w)


    def _handle_ae(self, ae):
        return torch.softmax(ae * 10 * self.gamma1, dim=1)
        
    def _filter_w(self, w, t):
        tmp_w = torch.softmax(w * 100 * self.gamma2, dim=0)
        # tmp_w[tmp_w < t] = 0
        return tmp_w

    def forward(self, x, train=True):
        bert_x, mask = self.bert(x, train)
        causal_x = self.cl(bert_x, x)
        return self.out(causal_x, self.bert.embedding.token.weight), mask
