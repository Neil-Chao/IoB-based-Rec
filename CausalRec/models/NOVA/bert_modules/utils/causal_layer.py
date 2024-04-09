import torch.nn as nn
import torch
import torch.nn.functional as F


class CausalLayer(nn.Module):

    def __init__(self, ae, w):
        super(CausalLayer, self).__init__()
        self.ae = ae
        self.w = w
        self.cluster_num = w.shape[0]

    def forward(self, bert_x, x):
        res = bert_x.clone()
        for j in range(x.shape[1]):
            target = x[:, j]
            for i in range(j):
                source = x[:, i]
                fac = (self.ae[source].unsqueeze(1) @ self.w @ self.ae[target].unsqueeze(2)).squeeze()
                res[:, j] += bert_x[:, i] * (( (i+1) / (j+1) ) * fac).view(-1, 1)
        return res
    
    