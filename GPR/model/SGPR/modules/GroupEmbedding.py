import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
import math
import functools
import random
import json
    

class GroupEmbedding(nn.Module):
    
    def __init__(self, user_num, item_num, user_emb_size, item_emb_size, factor=0.5) -> None:
        super().__init__()
        self.user_emb = nn.Embedding(user_num, user_emb_size, padding_idx=0)
        self.item_emb = nn.Embedding(item_num, item_emb_size, padding_idx=0)
        self.item_emb_size = item_emb_size
        self.user_emb_size = user_emb_size
        self.factor = factor
        
    def forward(self, group_user, group_behavior, target_user, similarity_vec):
        '''
        - group_user: n
        - group_behavior: n * m 
        
        n是用户总数
        m是行为总数
        '''
        group_embedding = torch.Tensor().cuda()
        for users, user_behaviors, single_target in zip(group_user, group_behavior, target_user):
            loaded_users = torch.tensor(json.loads(users)).cuda()
            users_embedding = self.user_emb(loaded_users)
            loaded_user_behaviors = json.loads(user_behaviors)
            user_num = loaded_users.shape[0]
            single_group_embedding = torch.zeros(user_num, self.item_emb_size).cuda()

            others_sim_vec = similarity_vec[loaded_users]
            target_sim_vec = similarity_vec[single_target]
            sim_values = self.factor * (target_sim_vec @ others_sim_vec.unsqueeze(-1))

            for i in range(user_num):
                single_user_behavior = loaded_user_behaviors[i]
                b_ids, count = torch.tensor(list(map(int, single_user_behavior.keys()))).cuda(), torch.tensor(list(single_user_behavior.values())).cuda()
                single_user_behavior_embedding = torch.sum(self.item_emb(b_ids) * count.unsqueeze(-1), dim=0)
                single_personalized_behavior_embedding = single_user_behavior_embedding * users_embedding[i]
                single_group_embedding[i] = single_personalized_behavior_embedding

            single_group_embedding = single_group_embedding * sim_values
            group_embedding = torch.cat((group_embedding, torch.sum(single_group_embedding, dim=0).unsqueeze(0)))

        return group_embedding
    


if __name__ == "__main__":
    d = {
        1: 95, 2: 85, 3: 59 
    }
    print(json.dumps(d))
    
