import torch.nn as nn
import torch
from .gelu import GELU
import torch.nn.functional as F

class OutLayer(nn.Module):

    def __init__(self, emb_size, num_items):
        super(OutLayer, self).__init__()
        self.W = nn.Linear(emb_size, emb_size, bias=True)
        self.b = nn.Parameter(torch.zeros(num_items, dtype=torch.float32, requires_grad=True))
        self.gelu = GELU()
        self.reset_parameters()

    def forward(self, x, E):
        return F.softmax(self.gelu(self.W(x)) @ E.T, dim=-1)
    
    @torch.no_grad()
    def reset_parameters(self):
        nn.init.uniform_(self.b, -0.02, 0.02)
        nn.init.uniform_(self.W.weight, -0.02, 0.02)