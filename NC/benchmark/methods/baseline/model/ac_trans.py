import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HGNN_AC(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout, activation, num_heads, rate, cuda=False):
        super(HGNN_AC, self).__init__()
        self.dropout = dropout
        self.attentions = [SelfAttentionLayer(in_dim, hidden_dim, dropout, activation, rate, cuda) for _ in range(num_heads)]
        
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, emb_dest, emb_src, feat_src):
        x = torch.cat([att(emb_dest, emb_src, feat_src).unsqueeze(0) for att in self.attentions], dim=0)
        return torch.mean(x, dim=0, keepdim=False)

class SelfAttentionLayer(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout, activation, rate, cuda=False):
        super(SelfAttentionLayer, self).__init__()
        self.dropout = dropout
        self.activation = activation
        self.is_cuda = cuda
        self.rate = rate
        self.W_query = nn.Linear(in_dim, hidden_dim)
        self.W_key = nn.Linear(in_dim, hidden_dim)
        self.W_value = nn.Linear(hidden_dim, hidden_dim)  # 修改这一行

        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, emb_dest, emb_src, feature_src):
        query = self.W_query(emb_dest)
        key = self.W_key(emb_src)
        value = self.W_value(feature_src)  # 使用 hidden_dim 而不是 in_dim

        attention_weights = torch.matmul(query, key.transpose(-2, -1))
        attention_weights = F.softmax(attention_weights / np.sqrt(query.size(-1)), dim=-1)
        attention_weights = F.dropout(attention_weights, p=self.dropout, training=self.training)

        out = torch.matmul(attention_weights, value)
        return self.activation(out)

