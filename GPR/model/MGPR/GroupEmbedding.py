import torch
import torch.nn as nn
import json

class GroupEmbedding(nn.Module):
    
    def __init__(self, user_num, item_num, user_emb_size, item_emb_size, pooling) -> None:
        super().__init__()
        self.user_emb = nn.Embedding(user_num, user_emb_size, padding_idx=0)
        
        self.linear = nn.Linear(item_emb_size, user_emb_size)
        self.item_emb_size = item_emb_size
        self.user_emb_size = user_emb_size
        self.pooling = pooling
        
    def forward(self, group_user, group_behavior, item_emb):
        '''
        - group_user: n
        - group_behavior: n * m 
        
        n是用户总数
        m是行为总数
        '''

        '''
        - group_user: n
        - group_behavior: n * m 
        
        n是用户总数
        m是行为总数
        '''
        group_embedding = torch.Tensor().cuda()
        for users, user_behaviors in zip(group_user, group_behavior):
            loaded_users = torch.tensor(json.loads(users)).cuda()
            users_embedding = self.user_emb(loaded_users)
            loaded_user_behaviors = json.loads(user_behaviors)
            user_num = loaded_users.shape[0]
            single_group_embedding = torch.zeros(user_num, self.item_emb_size).cuda()
            for i in range(user_num):
                single_user_behavior = loaded_user_behaviors[i]
                b_ids, count = torch.tensor(list(map(int, single_user_behavior.keys()))).cuda(), torch.tensor(list(single_user_behavior.values())).cuda()
                single_user_behavior_embedding = torch.sum(item_emb(b_ids) * count.unsqueeze(-1), dim=0)
                single_personalized_behavior_embedding = single_user_behavior_embedding * users_embedding[i]
                single_group_embedding[i] = single_personalized_behavior_embedding
            if self.pooling == "sum":
                pooling_embedding = torch.sum(single_group_embedding, dim=0).unsqueeze(0)
            else:
                pooling_embedding = torch.mean(single_group_embedding, dim=0).unsqueeze(0)
            group_embedding = torch.cat((group_embedding, pooling_embedding))
        return group_embedding