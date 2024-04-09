import torch.nn as nn
import torch

from .attention import MultiHeadedAttention
from .utils import SublayerConnection, PositionwiseFeedForward


class TransformerBlock(nn.Module):


#     def __init__(self, max_len, hidden, attn_heads, dropout):
#         super().__init__()
#         self.transformers = nn.ModuleList(
#             [Transformer(hidden, attn_heads, hidden * 4, dropout) for _ in range(max_len)])
        

#     def forward(self, x, mask):
#         # return torch.concat([self.transformers[i].forward(x[:, i, :], mask[:, :, i, i]) for i in range(len(self.transformers))], dim=1)
#         return torch.concat([self.transformers[i].forward(x, mask) for i in range(len(self.transformers))], dim=1)

# class Transformer(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)
