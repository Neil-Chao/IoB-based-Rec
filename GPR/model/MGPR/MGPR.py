import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

import torch
import torch.nn as nn
import numpy as np
import random
import torch.backends.cudnn as cudnn

from GPR.model.MGPR.bert_modules.bert import BERT
from GPR.model.MGPR.bert_modules.utils.outlayer import OutLayer

from GPR.model.MGPR.GroupEmbedding import GroupEmbedding


def fix_random_seed_as(random_seed):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

class MGPR(nn.Module):
    '''
    bert_max_len: Length of sequence for bert
    num_items: Number of total items
    bert_num_blocks: Number of transformer layers
    bert_num_heads: Number of heads for multi-attention
    bert_hidden_units: Size of hidden vectors (d_model)
    bert_dropout: Dropout probability to use throughout the model
    '''
    def __init__(self, user_num, item_num, user_emb_size, item_emb_size, bert_max_len=6, bert_num_blocks=2, bert_num_heads=4, bert_dropout=0.5, alpha=1e-3, pooling="sum") -> None:
        super().__init__()
        # fix_random_seed_as(0)
        self.group_layer = GroupEmbedding(user_num+1, item_num+1, user_emb_size, item_emb_size, pooling)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(3 * item_emb_size, 2 * item_emb_size), 
            nn.Linear(2 * item_emb_size, 1 * item_emb_size)])
        self.reduceLinear = nn.Linear(user_emb_size + item_emb_size, item_emb_size)
        hidden_size = user_emb_size
        self.memory_model = BERT(bert_max_len, item_num+1, bert_num_blocks, bert_num_heads, hidden_size, bert_dropout)
        self.alpha = alpha
        self.dropout = nn.Dropout(bert_dropout)
        self.out = OutLayer(hidden_size, item_num + 1)

    def forward(self, group_user, group_behavior, x, target_user, train=False):

        group_embedding = self.group_layer(group_user, group_behavior, self.memory_model.embedding.token)

        user_embedding = self.group_layer.user_emb(target_user)
        
        
        h, mask = self.memory_model(x, train)
        h = h[:, -1, :]
        # dim有待测试
        g_u = torch.concat([group_embedding, user_embedding], dim=1)
        # 可能影响推荐性能
        g_u = self.reduceLinear(g_u)
        # pooling
        
        
        x = torch.concat([g_u * h, g_u, h], dim=1)
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(self.dropout(x))
        res = h + self.alpha * x

        return self.out(res, self.memory_model.embedding.token.weight), mask
    
    def pretrain(self, group_user, group_behavior, x, target_user, train=False):
        
        h, mask = self.memory_model(x, train)

        return self.out(h, self.memory_model.embedding.token.weight), mask
        


def loss_fn(mask, y_hat, y):
    '''
    mask: B * S
    y_hat: B * S * (d+1)
    y: B * S
    '''
    i_pos = []
    j_pos = []
    k_pos = []
    for i in range(mask.size(0)):
        for j in range(mask.size(1)):
            if mask[i][j] == 0:
                i_pos.append(i)
                j_pos.append(j)
                k_pos.append(y[i][j])
    return torch.sum(-torch.log(y_hat[i_pos, j_pos, k_pos])) / len(i_pos)


if __name__ == "__main__":
    pass