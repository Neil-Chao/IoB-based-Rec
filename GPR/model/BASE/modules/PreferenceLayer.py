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



class PreferenceLayer(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
        
        
    def forward(self, group_user, group_behavior):
        '''
        - group_user: n
        - group_behavior: n * m 
        
        n是用户总数
        m是行为总数
        '''


        return