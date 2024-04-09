import torch
import torch.nn as nn

class RUMC(nn.Module):
    
    def __init__(self, item_num, user_num, emb_size, ae, w, t=0.3, gamma1=5, gamma2=5, alpha=0.2, beta=1, k=0.1) -> None:
        super().__init__()
        self.item_emb = nn.Embedding(item_num + 1, emb_size, padding_idx=0)
        self.user_emb = nn.Embedding(user_num, emb_size)
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.ae = self._handle_ae(ae)
        self.w = self._filter_w(w, t)
        self.beta = beta
        self.alpha = alpha
        self.item_num = item_num
        self.user_num = user_num
        self.cluster_num = self.w.shape[0]
        self.k=k

        self.time_attenuate_func()
        
        

    def _handle_ae(self, ae):
        return torch.softmax(ae * 10 * self.gamma1, dim=1)
        
    def _filter_w(self, w, t):
        tmp_w = torch.softmax(w * 100 * self.gamma2, dim=0)
        # tmp_w[tmp_w < t] = 0
        return tmp_w
    
    def time_attenuate_func(self):
        tmp = torch.ones(5).cuda()
        for i in range(5):
            tmp[i] = 1 - (4-i) * self.k

        self.time_factor = tmp
        # self.time_factor = torch.softmax(tmp, dim=-1)

    def forward(self, u, X, y):
        batch_size = X.size(0)
        u_emb = self.user_emb(u.to(torch.int)).to(torch.float64)
        X_emb = self.item_emb(X.to(torch.int)).to(torch.float64)
        y_emb = self.item_emb(y.to(torch.int)).to(torch.float64)
        fac = (self.ae[X.flatten()].reshape(X.shape[0], X.shape[1], self.cluster_num) @ self.w @ self.ae[y.to(torch.int64)].unsqueeze(2)).squeeze()
        time_factor = self.time_factor.unsqueeze(0).repeat(batch_size, 1)
        w = (X_emb @ y_emb.unsqueeze(2)).squeeze() * self.beta * (torch.e ** fac) * time_factor
        z = torch.softmax(w, dim=1)
        p = torch.sum(X_emb * z.unsqueeze(2), dim=1)
        p_u = p * self.alpha + u_emb
        y_hat = torch.diag(torch.sigmoid(p_u @ y_emb.T))
        return y_hat