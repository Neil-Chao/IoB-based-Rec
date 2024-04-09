import torch
import torch.nn as nn

class GRU4Rec(nn.Module):
    def __init__(self, input_size, emb_size, num_layers) -> None:
        super().__init__()
        self.input_size = input_size
        self.emb_size = emb_size
        self.emb = nn.Embedding(input_size + 1, emb_size, padding_idx=0)
        self.gru = nn.GRU(
            batch_first=True,
            input_size=emb_size,
            hidden_size=emb_size,
            num_layers=num_layers,
            dropout=0.5
        )
        self.final_activation = nn.Tanh()
        self.feedforward = nn.Linear(emb_size, input_size + 1)

    def forward(self, X):
        X_emb = self.emb(X.to(torch.int))
        output, hn = self.gru(X_emb)
        user_emb = output[:, -1]
        res = self.final_activation(self.feedforward(user_emb))
        return res
    

