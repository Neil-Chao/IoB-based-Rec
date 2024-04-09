from torch.utils.data import Dataset
import torch
import json

class GroupDataset(Dataset):
    def __init__(self, dataset, user_num, behavior_num):
        self.dataset = dataset
        self.user_num = user_num
        self.behavior_num = behavior_num
        
    def __getitem__(self, index):
        x = self.dataset[index]
        group_user = json.loads(x[0])
        group_behavior = json.loads(x[1])
        group_user_emb = torch.zeros(self.user_num)
        group_behavior_emb = torch.zeros([self.user_num, self.behavior_num])
        
        for i in range(len(group_user)):
            user = group_user[i]
            behavior = group_behavior[i]
            group_user_emb[int(user)] = 1
            for k, v in behavior.items():
                group_behavior_emb[int(user)][int(k)] = v
        return (group_user_emb, group_behavior_emb, x[2], x[3], x[4])
    
    def __len__(self):
        return len(self.dataset)

class UnloadedGroupDatasetWithUserHistory(Dataset):
    def __init__(self, dataset, user_num, behavior_num):
        self.dataset = dataset
        self.user_num = user_num
        self.behavior_num = behavior_num

    def __getitem__(self, index):
        x = self.dataset[index]
        group_user = json.loads(x[0])
        group_behavior = json.loads(x[1])
        group_user_emb = torch.zeros(self.user_num)
        group_behavior_emb = torch.zeros([self.user_num, self.behavior_num])
        
        for i in range(len(group_user)):
            user = group_user[i]
            behavior = group_behavior[i]
            group_user_emb[int(user)] = 1
            for k, v in behavior.items():
                group_behavior_emb[int(user)][int(k)] = v
        return (group_user_emb, group_behavior_emb, x[3], x[4], x[5])
    
    def __len__(self):
        return len(self.dataset)