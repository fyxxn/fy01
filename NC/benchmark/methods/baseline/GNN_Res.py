import torch
import torch.nn as nn
import dgl
from dgl.nn.pytorch import GraphConv
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax, GATConv
from conv import myGATConv

class myGAT1(nn.Module):
    def __init__(self,
                 g,
                 edge_dim,
                 num_etypes,
                 in_dims,
                 num_hidden,
                 num_classes,
                 num_layers,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 alpha):
        super(myGAT1, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        self.fc_res = nn.Linear(num_hidden, num_hidden, bias=True)
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc_res.weight, gain=1.414)
        # input projection (no residual)
        self.gat_layers.append(myGATConv(edge_dim, num_etypes,
            2*num_hidden, 2*num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation, alpha=alpha))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(myGATConv(edge_dim, num_etypes,
                2*num_hidden * heads[l-1], 2*num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation, alpha=alpha))
        # output projection
        self.gat_layers.append(myGATConv(edge_dim, num_etypes,
            2*num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None, alpha=alpha))
        self.epsilon = torch.FloatTensor([1e-12]).cuda()

    def forward(self, features_list, e_feat):
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            temp = fc(feature)
            # h.append(torch.nn.functional.relu(fc(feature)))
            # 加
            # h.append(F.relu(self.fc_res(temp))+F.relu(temp))
            # 拼接
            h.append(torch.cat((F.relu(self.fc_res(temp)),F.relu(temp)),1))

        h = torch.cat(h, 0)
        res_attn = None
        for l in range(self.num_layers):
            h, res_attn = self.gat_layers[l](self.g, h, e_feat, res_attn=res_attn)
            h = h.flatten(1)
        # output projection
        logits, _ = self.gat_layers[-1](self.g, h, e_feat, res_attn=None)
        logits = logits.mean(1)
        # This is an equivalent replacement for tf.l2_normalize, see https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/math/l2_normalize for more information.
        logits = logits / (torch.max(torch.norm(logits, dim=1, keepdim=True), self.epsilon))
   
        return logits

