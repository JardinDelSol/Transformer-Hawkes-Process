import torch
import numpy as np
import torch.nn as nn
from transformer.Modules import ScaledDotProductAttention
from transformer.SubLayers import MultiHeadAttention, PositionwiseFeedForward
import math


class AttnBlock(nn.Module):
    def __init__(self):
        super(AttnBlock, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head=4,
            d_model=512,
            d_k=512,
            d_v=512,
            dropout=0.1,
            normalize_before=False,
        )
        self.pos_ffn = PositionwiseFeedForward(512, 1024, 0.1, False)

    def forward(self, X, pad_mask, attn_mask):
        assert X.shape[1] == attn_mask.shape[-1] and X.shape[1] == attn_mask.shape[-2]
        S, attn = self.slf_attn(X, X, X, mask=attn_mask)
        S = S * pad_mask.unsqueeze(-1)

        H = self.pos_ffn(S)
        H = H * pad_mask.unsqueeze(-1)

        return H


class parameters(nn.Module):
    def __init__(self):
        self.alpha = nn.parameter(torch.tensor(0.5))
        self.beta = nn.parameter(torch.tensor(1.0))


def get_padd_mask(data):
    # mask = torch.where(data == 0, torch.ones_like(data), torch.zeros_like(data))
    # return mask
    return data.ne(0).type(torch.float)

class Transformer(nn.Module):
    def __init__(self, B, D, embed_dim, num_attention, device):
        super(Transformer, self).__init__()

        self.B = B
        self.D = D
        self.T = None
        self.embed_dim = embed_dim
        self.num_attention = num_attention
        self.device = device

        self.event_embedding = nn.Embedding(self.D + 1, self.embed_dim, padding_idx=0)

        AttnBlocks = []
        for _ in range(num_attention):
            AttnBlocks.append(AttnBlock())

        self.AttnBlocks = nn.ModuleList(AttnBlocks)

        self.time_pred = nn.Linear(self.embed_dim, 1)
        nn.init.xavier_normal_(self.time_pred.weight)
        self.event_pred = nn.Linear(self.embed_dim, self.D)
        nn.init.xavier_normal_(self.event_pred.weight)

        self.linear = nn.Linear(self.embed_dim, self.D)

        ###
        self.num_types = self.D

        ### Params
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, event, time):
        """
        time: B, T
        event: B, T
        """
        self.T = time.shape[-1]
        time_mask = self.create_time_mask().to(self.device)  # B, T, T
        attn_pad_mask = self.create_attn_mask(event).to(self.device)  # B, T, T
        attn_pad_mask = (time_mask + attn_pad_mask) > 0

        pad_mask = get_padd_mask(event)  # B, T

        Z = self.time_embedding(time)  # B, T, enc_dim
        Z = Z * pad_mask.unsqueeze(-1)  # B, T, enc_dim

        UY = self.event_embedding(event)  # B, T, enc_dim
        UY = UY * pad_mask.unsqueeze(2)  # B, T, enc_dim
        X = Z + UY  # B, T, enc_dim
        H = X  # B, T, enc_dim

        for attn in self.AttnBlocks:
            H = attn(H, pad_mask, attn_pad_mask)  # B, T, enc_dim
            H += Z  # B, T, enc_dim

        H -= Z

        pred_t = self.time_pred(H) * pad_mask.unsqueeze(-1)  # 1
        pred_e = self.event_pred(H) * pad_mask.unsqueeze(-1)  # D

        return H, (pred_e, pred_t)

    def create_time_mask(self):
        temp = torch.ones((self.T, self.T))
        triu = torch.triu(temp, diagonal=1)
        mask = triu.unsqueeze(0).expand(self.B, -1, -1)
        return mask

    def create_attn_mask(self, event):
        mask = event == 0
        attn_mask = mask.unsqueeze(1).expand(-1, self.T, -1).float()
        return attn_mask

    def time_embedding(self, time):
        def get_denom():
            temp = []
            for i in range(self.embed_dim):
                temp.append(math.pow(1e4, (i - i % 2) / self.embed_dim))
            return torch.tensor(temp).to(self.device)

        denom = get_denom()

        temp = time.unsqueeze(2) / denom  # B, T, enc_dim
        even = temp[:, :, 0:-1:2]
        odd = temp[:, :, 1:-1:2]
        even = torch.sin(even)
        odd = torch.cos(odd)
        temp[:, :, 0:-1:2] = even
        temp[:, :, 1:-1:2] = odd

        return temp

