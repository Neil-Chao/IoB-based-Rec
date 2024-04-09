from torch.utils.data import Dataset
import torch
import json
import numpy as np

class GroupDatasetWithUserHistory(Dataset):
    def __init__(self, dataset, user_num, behavior_num):
        self.dataset = dataset
        self.user_num = user_num
        self.behavior_num = behavior_num
        
    def __getitem__(self, index):
        x = self.dataset[index]
        group_user = json.loads(x[0])
        group_behavior = json.loads(x[1])
        user_behavior = json.loads(x[2])
        group_user_emb = torch.zeros(self.user_num)
        group_behavior_emb = torch.zeros([self.user_num, self.behavior_num])
        
        for i in range(len(group_user)):
            user = group_user[i]
            behavior = group_behavior[i]
            group_user_emb[int(user)] = 1
            for k, v in behavior.items():
                group_behavior_emb[int(user)][int(k)] = v
        return (group_user_emb, group_behavior_emb, torch.tensor(user_behavior), x[3], x[4], x[5])
    
    def __len__(self):
        return len(self.dataset)
    

class UnloadedGroupDatasetWithUserHistory(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __getitem__(self, index):
        x = self.dataset[index]
        user_behavior = json.loads(x[2])
        return (x[0], x[1], torch.tensor(user_behavior), x[3], x[4], x[5])
    
    def __len__(self):
        return len(self.dataset)
    
class UnloadedGroupDatasetForOthers(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __getitem__(self, index):
        x = self.dataset[index]
        user_behavior = json.loads(x[2])
        return (x[0], x[1], torch.tensor(user_behavior), x[3], x[4], x[5])
    
    def __len__(self):
        return len(self.dataset)
    
class UnloadedGroupDatasetWithNegative(Dataset):
    def __init__(self, dataset, item_num):
        self.dataset = dataset
        self.item_num = item_num

    def negative_sampling(self, X, y, count=1):
        interacted = np.append(X, [y, 0])
        items = np.arange(0, self.item_num + 1)
        selected = np.delete(items, interacted)
        return np.random.choice(selected, count, False)[0]
        
    def __getitem__(self, index):
        x = self.dataset[index]
        user_behavior = json.loads(x[2])
        neg = self.negative_sampling(user_behavior, x[4])
        return (x[0], x[1], torch.tensor(user_behavior + [x[4]]), x[3], x[4], neg)
    
    def __len__(self):
        return len(self.dataset)