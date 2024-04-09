from torch.utils.data import Dataset
import torch
import numpy as np
import json

class GroupCFDatasetForOthers(Dataset):
    def __init__(self, dataset, group_item_num, user_num):
        self.dataset = dataset
        self.group_item_num = group_item_num
        self.user_num = user_num
        
    def __getitem__(self, index):
        x = self.dataset[index]
        items = np.arange(self.group_item_num)
        selected = np.delete(items, x[1])
        negative = np.random.choice(selected, 1, False)[0]
        group = torch.zeros(self.user_num)
        group_arr = json.loads(x[0])
        for e in group_arr:
            group[e] = 1
        return (group, x[1], negative)
    
    def __len__(self):
        return len(self.dataset)
    
class GroupCFDataset(Dataset):
    def __init__(self, dataset, group_item_num, user_num):
        self.dataset = dataset
        self.group_item_num = group_item_num
        self.user_num = user_num
        
    def __getitem__(self, index):
        x = self.dataset[index]
        items = np.arange(self.group_item_num)
        selected = np.delete(items, x[1])
        negative = np.random.choice(selected, 1, False)[0]
        group = torch.zeros(self.user_num)
        for e in x[0]:
            group[e] = 1
        return (group, x[1], negative)
    
    def __len__(self):
        return len(self.dataset)
    
class GroupCFValDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        x = self.dataset[index]
        return (int(x[0]), int(x[1]), int(x[2]))
    
    def __len__(self):
        return len(self.dataset)