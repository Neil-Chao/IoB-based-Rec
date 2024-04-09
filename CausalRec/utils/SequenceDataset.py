from torch.utils.data import Dataset
import numpy as np
import torch
import faiss
import json

class SequenceDataset(Dataset):
    def __init__(self, dataset, item_num):
        self.dataset = dataset
        self.item_num = item_num

    def negative_sampling(self, X, y, count=1):
        interacted = np.append(X, y)
        items = np.arange(self.item_num)
        selected = np.delete(items, interacted)
        return np.random.choice(selected, count, False)[0]
        
    def __getitem__(self, index):
        x = self.dataset[index]
        neg = self.negative_sampling(x[1], x[2])
        return (x[0], torch.tensor(x[1]), x[2], neg)
    
    def __len__(self):
        return len(self.dataset)
    
class BERTDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __getitem__(self, index):
        x = self.dataset[index]
        return (x[0], torch.tensor(x[1]))
    
    def __len__(self):
        return len(self.dataset)
    
class NOVADataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __getitem__(self, index):
        x = self.dataset[index]
        return (x[0], torch.tensor(x[1]), torch.tensor(x[2]), torch.tensor(x[3]))
    
    def __len__(self):
        return len(self.dataset)

class MrTransformerDataset(Dataset):
    def __init__(self, dataset, vocab_size, config):
        self.dataset = dataset
                
    

    def __getitem__(self, index):
        x = self.dataset[index]
        return (torch.tensor(x[1]), torch.tensor(x[2]), torch.tensor(x[3]), torch.tensor(x[4]))
    
    def __len__(self):
        return len(self.dataset)
    
class UnloadedGroupDatasetForOthers(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __getitem__(self, index):
        x = self.dataset[index]
        user_behavior = json.loads(x[2])
        return (x[3], torch.tensor(user_behavior), x[4], x[5])
    
    def __len__(self):
        return len(self.dataset)
    
class UnloadedGroupDatasetABC(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __getitem__(self, index):
        x = self.dataset[index]
        user_behavior = json.loads(x[2])
        return (x[0], x[1], x[3], torch.tensor(user_behavior), x[4], x[5])
    
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
        return (x[3], torch.tensor(user_behavior), x[4], neg)
    
    def __len__(self):
        return len(self.dataset)
    
class UnloadedGroupDatasetForBERT(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        
        
    def __getitem__(self, index):
        x = self.dataset[index]
        user_behavior = json.loads(x[2])
        return (x[3], torch.tensor(user_behavior + [x[4]]))
    
    def __len__(self):
        return len(self.dataset)
