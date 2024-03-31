import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class HGNN_AC(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout, activation, num_heads, rate,cuda=False):
        super(HGNN_AC, self).__init__()
        self.dropout = dropout
        self.attentions = [AttentionLayer(in_dim, hidden_dim, dropout, activation, rate,cuda) for _ in range(num_heads)]
        
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, emb_dest, emb_src, feat_src):
        x = torch.cat([att(emb_dest, emb_src, feat_src).unsqueeze(0) for att in self.attentions], dim=0)

        return torch.mean(x, dim=0, keepdim=False)


class AttentionLayer(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout, activation, rate,cuda=False):
        super(AttentionLayer, self).__init__()
        self.dropout = dropout
        self.activation = activation
        self.is_cuda = cuda
        self.rate=rate
        self.W = nn.Parameter(nn.init.xavier_normal_(
            torch.Tensor(in_dim, hidden_dim).type(torch.cuda.FloatTensor if cuda else torch.FloatTensor),
            gain=np.sqrt(2.0)), requires_grad=True)
        self.W2 = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(hidden_dim, hidden_dim).type(
            torch.cuda.FloatTensor if cuda else torch.FloatTensor), gain=np.sqrt(2.0)),
            requires_grad=True)

        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, emb_dest, emb_src, feature_src):
        h_1 = torch.mm(emb_src, self.W)
        h_2 = torch.mm(emb_dest, self.W)

        e = self.leakyrelu(torch.mm(torch.mm(h_2, self.W2), h_1.t()))
        #zero_vec = -9e15 * torch.ones_like(e)
        # print(emb_dest.shape)
        # print(emb_src.shape)
        # print(feature_src.shape)
        # print(e.shape)
        # print(self.rate*0.01*emb_src.shape[0])
        #----------------------------------------------------------------------------------------------------
        #----------------------------------------------------------------------------------------------------
        #sorted, indices = torch.sort(e,descending=False)
        # indices = indices[:,:int(self.rate*0.01*emb_src.shape[0])]
        # temp = torch.zeros_like(e)
        # for i in range(e.shape[0]):
        #     temp[i,indices[i]]=1
        # zero_vec = -9e15 * torch.ones_like(e)
        # attention = torch.where(temp > 0, e, zero_vec)
        # attention = F.softmax(attention, dim=1)
        #----------------------------------------------------------------------------------------------------
        #----------------------------------------------------------------------------------------------------
        attention = F.softmax(e, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, feature_src)
        # attention = F.softmax(h_prime, dim=1)
        return self.activation(h_prime)
