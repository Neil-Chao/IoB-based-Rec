import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
import math
import functools
import random
import json



from GPR.model.SGPR.modules.GroupEmbedding import GroupEmbedding
from GPR.model.SGPR.modules.PreferenceLayer import PreferenceLayer

    
class SGPR(nn.Module):
    '''
    Attentive Group Personalized Recommendation
    '''
    def __init__(self, user_num, item_num, user_emb_size, item_emb_size, similarity_vec, factor=0.5, drop_prob=0.5) -> None:
        super().__init__()
        self.group_layer = GroupEmbedding(user_num+1, item_num+1, user_emb_size, item_emb_size, factor)
        self.preference_layer = PreferenceLayer()
        self.hidden_layers = nn.ModuleList([
            nn.Linear(3 * item_emb_size, 2 * item_emb_size), 
            nn.Linear(2 * item_emb_size, 1 * item_emb_size), 
            nn.Linear(1 * item_emb_size, 1)])
        self.reduceLinear = nn.Linear(user_emb_size + item_emb_size, item_emb_size)
        self.dropout = nn.Dropout(drop_prob)
        self.similarity_vec = similarity_vec
        self.user_emb_size = user_emb_size


    def forward(self, group_user, group_behavior, target_user, target_item):
        item_embedding = self.group_layer.item_emb(target_item)
        
        group_embedding = self.group_layer(group_user, group_behavior, target_user, self.similarity_vec)
        # preference_embedding = torch.zeros(self.user_emb_size).to(torch.float).cuda()
        # target_user = target_user.squeeze()
        # target_item = target_item.squeeze()
        user_embedding = self.group_layer.user_emb(target_user)
        
        # dim有待测试
        g_u = torch.concat([group_embedding, user_embedding], dim=1)
        # 可能影响性能
        g_u = self.reduceLinear(g_u)
        # pooling
        x = torch.concat([g_u * item_embedding, g_u, item_embedding], dim=1)
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(self.dropout(x))
        return x
    
    def predict(self, group_user, group_behavior, target_user):
        items_embedding = self.group_layer.item_emb.weight.data
        
        group_embedding = self.group_layer(group_user, group_behavior, target_user, self.similarity_vec)
        # preference_embedding = torch.zeros(self.user_emb_size).to(torch.float).cuda()
        # target_user = target_user.squeeze()
        # target_item = target_item.squeeze()
        user_embedding = self.group_layer.user_emb(target_user)
        
        # dim有待测试
        g_u = torch.concat([group_embedding, user_embedding], dim=1)
        # 可能影响性能
        g_u = self.reduceLinear(g_u)
        g_u = g_u.unsqueeze(1).repeat(1, items_embedding.shape[0], 1)
        items_embedding = items_embedding.unsqueeze(0).repeat(g_u.shape[0], 1, 1)
        # pooling
        x = torch.concat([g_u * items_embedding, g_u, items_embedding], dim=-1)
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
        return x

    