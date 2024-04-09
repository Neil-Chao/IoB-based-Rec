import torch
import torch.nn as nn

class GRU4RecR(nn.Module):
    def __init__(self, input_size, emb_size, num_layers) -> None:
        super().__init__()
        cluster = 16
        self.ae = nn.Parameter(torch.zeros(input_size + 1, cluster).to(torch.float64), requires_grad=True)
        self.w = nn.Parameter(torch.zeros(cluster, cluster).to(torch.float64), requires_grad=True)
        self.input_size = input_size
        self.emb_size = emb_size
        self.emb = nn.Embedding(input_size + 1, emb_size, padding_idx=0)
        self.gru = nn.GRU(
            batch_first=True,
            input_size=emb_size,
            hidden_size=emb_size,
            num_layers=num_layers,
            dropout=0.5
            # time_major=False,
        )
        self.final_activation = nn.Tanh()
        self.feedforward = nn.Linear(emb_size, input_size + 1, dtype=torch.float64)
        self.cluster_num = self.w.shape[0]
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        # k = 1.0 / self.input_size
        # bound = math.sqrt(k)
        bound = 0.05
        nn.init.uniform_(self.ae, -bound, bound)
        nn.init.uniform_(self.w, -bound, bound)

    def forward(self, X, y):
        X_emb = self.emb(X.to(torch.int))
        output, hn = self.gru(X_emb)
        user_emb = output[:, -1]
        fac = (self.ae[X.flatten()].reshape(X.shape[0], X.shape[1], self.cluster_num) @ self.w @ self.ae[y.to(torch.int64)].unsqueeze(2)).squeeze()
        for i in range(X.size(1) - 1):
            user_emb = user_emb + output[:, i] * ((i+1) / X.size(0) / 2) * fac[:, i].unsqueeze(1)
        res = self.final_activation(self.feedforward(user_emb))
        return res
    
    def predict(self, X):
        X_emb = self.emb(X.to(torch.int))
        output, hn = self.gru(X_emb)
        user_emb = output[:, -1]
        res = self.final_activation(self.feedforward(user_emb))
        return res