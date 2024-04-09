import torch
import torch.nn as nn

class GRU4RecC(nn.Module):
    def __init__(self, input_size, emb_size, num_layers, ae, w, t=0.3, gamma1=5, gamma2=5) -> None:
        super().__init__()
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.ae = self._handle_ae(ae)
        self.w = self._filter_w(w, t)
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

    def _handle_ae(self, ae):
        return torch.softmax(ae * 10 * self.gamma1, dim=1)
        
    def _filter_w(self, w, t):
        tmp_w = torch.softmax(w * 100 * self.gamma2, dim=0)
        # tmp_w[tmp_w < t] = 0
        return tmp_w

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