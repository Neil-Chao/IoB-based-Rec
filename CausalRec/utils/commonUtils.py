import torch
import pandas as pd
import random
import numpy as np

def get_causal(config):
    if config.scenario == config.MUSIC_INSTRUMENT:
        w_df = pd.read_csv("dataset/Musical_Instruments/w.csv", header=None)
        w = torch.tensor(w_df.to_numpy()).cuda()

        ae_df = pd.read_csv("dataset/Musical_Instruments/ae.csv", header=None)
        ae = torch.tensor(ae_df.to_numpy()).cuda()
    elif config.scenario == config.DIGITAL_MUSIC:
        w_df = pd.read_csv("dataset/Digital_Music/w.csv", header=None)
        w = torch.tensor(w_df.to_numpy()).cuda()

        ae_df = pd.read_csv("dataset/Digital_Music/ae.csv", header=None)
        ae = torch.tensor(ae_df.to_numpy()).cuda()
    elif config.scenario == config.LUXURY_BEAUTY:
        w_df = pd.read_csv("dataset/Luxury_Beauty/w.csv", header=None)
        w = torch.tensor(w_df.to_numpy()).cuda()

        ae_df = pd.read_csv("dataset/Luxury_Beauty/ae.csv", header=None)
        ae = torch.tensor(ae_df.to_numpy()).cuda()
    else:
        pass
    
    return w, ae

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)