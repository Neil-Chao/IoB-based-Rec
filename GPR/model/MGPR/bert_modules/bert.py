import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))
from torch import nn as nn
import torch
import numpy as np

from GPR.model.MGPR.bert_modules.embedding import BERTEmbedding
from GPR.model.MGPR.bert_modules.transformer import TransformerBlock
import torch.backends.cudnn as cudnn
import random

class BERT(nn.Module):
    def __init__(self, bert_max_len, num_items, bert_num_blocks, bert_num_heads, bert_hidden_units, bert_dropout):
        super().__init__()

        def fix_random_seed_as(random_seed):
            random.seed(random_seed)
            torch.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)
            np.random.seed(random_seed)
            cudnn.deterministic = True
            cudnn.benchmark = False
        fix_random_seed_as(0)

        max_len = bert_max_len
        num_items = num_items
        n_layers = bert_num_blocks
        heads = bert_num_heads
        # vocab_size = num_items + 2
        vocab_size = num_items
        hidden = bert_hidden_units
        self.hidden = hidden
        dropout = bert_dropout

        self.max_len = max_len
        self.num_items = num_items

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=self.hidden, max_len=max_len, dropout=dropout)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, heads, hidden * 4, dropout) for _ in range(n_layers)])
        # self.transformer_blocks = nn.ModuleList(
        #     [TransformerBlock(max_len, hidden, heads, dropout) for _ in range(n_layers)])

    def forward(self, x, train=False):
        if train:
            raw_mask = torch.tensor(np.array(([self.fill_zeros(np.random.choice(self.max_len, 2)) for _ in x]))).cuda()
        else:
            raw_mask = torch.ones_like(x)
            raw_mask[:, -1] = 0
        # torch.to
        # raw_mask = torch.rand_like(x.float()) > self.bert_mask_prob
        mask = raw_mask.unsqueeze(-1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x)
        x = torch.masked_fill(x, mask == 0, 0)
        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x)
        # for transformer_block in self.transformer_blocks:
        #     x = transformer_block.forward(x, mask)

        return x, raw_mask
    
    def fill_zeros(self, selected):
        res = np.ones(self.max_len)
        res[selected] = 0
        return res

    def init_weights(self):
        pass
