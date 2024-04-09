import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))
from torch import nn as nn
import torch
import numpy as np

from CausalRec.models.NOVA.bert_modules.embedding import BERTEmbedding
from CausalRec.models.NOVA.bert_modules.transformer import TransformerBlock
from CausalRec.models.NOVA.utils import fix_random_seed_as


class BERT(nn.Module):
    def __init__(self, args):
        super().__init__()

        fix_random_seed_as(args.model_init_seed)
        # self.init_weights()

        max_len = args.bert_max_len
        num_items = args.num_items
        n_layers = args.bert_num_blocks
        heads = args.bert_num_heads
        # vocab_size = num_items + 2
        vocab_size = num_items
        hidden = args.bert_hidden_units
        self.hidden = hidden
        dropout = args.bert_dropout

        self.max_len = max_len
        self.num_items = num_items
        self.bert_mask_num = int(args.bert_mask_prob * args.bert_max_len)

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=self.hidden, max_len=max_len, dropout=dropout)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, heads, hidden * 4, dropout) for _ in range(n_layers)])
        # self.transformer_blocks = nn.ModuleList(
        #     [TransformerBlock(max_len, hidden, heads, dropout) for _ in range(n_layers)])

    def forward(self, x, overall, timeDiff, train=True):
        # if train:
        #     raw_mask = torch.tensor(np.array(([self.fill_zeros(np.random.choice(self.max_len, 2)) for _ in x]))).cuda()
        # else:
        #     raw_mask = torch.ones_like(x)
        #     raw_mask[:, -1] = 0
        raw_mask = torch.ones(x.shape[0], x.shape[1])
        raw_mask[:, -1] = 0
        mask = raw_mask.unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1).cuda()

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x)
        behavior_embed = torch.concat([x, overall.unsqueeze(-1), timeDiff.unsqueeze(-1)], dim=-1)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, behavior_embed, mask)
        # for transformer_block in self.transformer_blocks:
        #     x = transformer_block.forward(x, mask)

        return x, raw_mask
    
    def fill_zeros(self, selected):
        res = np.ones(self.max_len)
        res[selected] = 0
        return res

    def init_weights(self):
        pass
