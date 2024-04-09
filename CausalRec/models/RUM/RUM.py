import torch
import torch.nn as nn

class RUM(nn.Module):
    
    def __init__(self, item_num, user_num, emb_size, alpha=0.2, beta=1) -> None:
        super().__init__()
        self.item_emb = nn.Embedding(item_num + 1, emb_size, padding_idx=0)
        self.user_emb = nn.Embedding(user_num, emb_size)
        self.beta = beta
        self.alpha = 1
        self.item_num = item_num
        self.user_num = user_num
        self.sigmoid = nn.Sigmoid()

    def forward(self, u, X, y):
        u_emb = self.user_emb(u.to(torch.int))
        X_emb = self.item_emb(X.to(torch.int))
        y_emb = self.item_emb(y.to(torch.int))
        w = (X_emb @ y_emb.unsqueeze(2)).squeeze() * self.beta
        z = torch.softmax(w, dim=1)
        p = torch.sum(X_emb * z.unsqueeze(2), dim=1)
        p_u = p * self.alpha + u_emb
        y_hat = torch.diag(self.sigmoid(p_u @ y_emb.T))
        return y_hat
    
    def predict(self, u, X):
        u_emb = self.user_emb(u.to(torch.int))
        X_emb = self.item_emb(X.to(torch.int))
        y_embs = self.item_emb.weight.data.unsqueeze(0).repeat(u.shape[0], 1, 1)
        w = torch.matmul(X_emb, y_embs.transpose(-1, -2)) * self.beta
        z = torch.softmax(w, dim=1).unsqueeze(-1)
        X_emb = X_emb.unsqueeze(2).repeat(1, 1, y_embs.shape[1], 1)
        p = torch.sum(X_emb * z, dim=1)
        u_emb = u_emb.unsqueeze(1).repeat(1, y_embs.shape[1], 1)
        p_u = (p * self.alpha + u_emb).transpose(0, 1)
        y_embs = y_embs.transpose(0, 1).transpose(1, 2)
        y_hat = torch.diagonal(self.sigmoid(p_u @ y_embs), dim1=-1, dim2=-2)
        return y_hat





    

    
