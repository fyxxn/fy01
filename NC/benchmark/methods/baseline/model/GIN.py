import torch
import torch.nn as nn
import dgl
from dgl.nn.pytorch import GraphConv
from dgl.nn import GINConv

import dgl.function as fn
# from dgl.function import function as fn

from dgl.nn.pytorch import edge_softmax, GATConv
from model.conv import myGATConv
from torch.nn.functional import sigmoid
import torch.nn.functional as F
import numpy as np
from model.ac import HGNN_AC
from model.ginmodule import GINClassifier

import torch as th
from torch import nn

from dgl.utils import expand_as_pair


class myGAT(nn.Module):
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
                 alpha,
                 emb_dim,
                 attn_vec_dim,
                 num_heads,rate,
                 dropout_rate=0.5,
                 cuda=False):
        super(myGAT, self).__init__()
        # print(heads)
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.hidden_dim = num_hidden
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        self.fc=nn.Linear(num_hidden, num_hidden, bias=True)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        self.bu_gat=myGATConv(edge_dim, num_etypes,
            num_hidden, num_hidden, 1,
            feat_drop, attn_drop, negative_slope, False, self.activation, alpha=alpha)
        
        # input projection (no residual)
        # self.gat_layers.append(myGATConv(edge_dim, num_etypes,
        #     num_hidden, num_hidden, heads[0],
        #     feat_drop, attn_drop, negative_slope, False, self.activation, alpha=alpha))
        lin1=torch.nn.Linear(num_hidden, num_hidden)
        self.gat_layers.append(myGINConv(lin1,'mean',activation=F.sigmoid))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            # self.gat_layers.append(myGATConv(edge_dim, num_etypes,
            #     num_hidden * heads[l-1], num_hidden, heads[l],
            #     feat_drop, attn_drop, negative_slope, residual, self.activation, alpha=alpha))
            temp=torch.nn.Linear(num_hidden, num_hidden)
            self.gat_layers.append(myGINConv(temp,'mean',activation=F.sigmoid))
        # output projection
        # self.gat_layers.append(myGATConv(edge_dim, num_etypes,
        #     num_hidden * heads[-2], num_classes, heads[-1],
        #     feat_drop, attn_drop, negative_slope, residual, None, alpha=alpha))
        lin2=torch.nn.Linear(num_hidden, num_classes)
        self.gat_layers.append(myGINConv(lin2,'mean',activation=F.sigmoid))
        self.epsilon = torch.FloatTensor([1e-12]).cuda()

        self.hgnn_ac = HGNN_AC(in_dim=emb_dim, hidden_dim=attn_vec_dim, dropout=dropout_rate,
                               activation=F.sigmoid, num_heads=num_heads,rate=rate, cuda=cuda)
        self.gin=GINClassifier(7, 3, num_hidden, num_hidden, num_classes, feat_drop,
                          False, 'sum', 'sum', feat_drop, cuda)

    def forward(self, features_list,e_feat,feat_drop_idx,node_type_src,mask_list,feat_keep_idx,device,emb,type_mask,args):
        
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        #----------------------------------------------------------------------------------------------
        feat_src = h

        
        # attribute completion
        
        if args.dataset=='IMDB' or args.dataset=='ACM':
            feature_src_re = self.hgnn_ac(emb, emb[mask_list[node_type_src]],
                                    feat_src[mask_list[node_type_src]])
            loss_ac=torch.tensor(0)
        else:
            feature_src_re = self.hgnn_ac(emb, emb[mask_list[node_type_src]][feat_keep_idx],
                                        feat_src[mask_list[node_type_src]][feat_keep_idx])
            loss_ac = F.mse_loss(feat_src[mask_list[node_type_src]][feat_drop_idx],
                            feature_src_re[mask_list[node_type_src]][feat_drop_idx],reduction='mean')
          


        if args.dataset=='DBLP':
            for i in range(len(features_list)):
            #     feat_src[mask_list[i]]+=feature_src_re[mask_list[i]]
                if i==node_type_src:          
                    feat_src[mask_list[i]]+=feature_src_re[mask_list[i]]
            # for i in range(len(features_list)):
            #     if i!=node_type_src:          
            #         feat_src[mask_list[i]]+=feature_src_re[mask_list[i]]
        else:
            for i in range(len(features_list)):
                if i!=node_type_src:          
                    feat_src[mask_list[i]]+=feature_src_re[mask_list[i]]
        
       
        if args.dataset=='DBLP':
            h=self.fc(feat_src)
        else:
            h=feat_src
        
        #----------------------------------------------------------------------------------------------
        res_attn = None
        for l in range(self.num_layers):
            # print(h.shape)
            # print(l)
            h = self.gat_layers[l](self.g, h)
            h = h.flatten(1)

        # output projection
        logits = self.gat_layers[-1](self.g, h)
        # # print(logits.shape)
        # logits = logits.unsqueeze(dim=1)
        # logits = logits.mean(1)
        # # print(logits.shape)
        # # This is an equivalent replacement for tf.l2_normalize, see https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/math/l2_normalize for more information.
        # logits = logits / (torch.max(torch.norm(logits, dim=1, keepdim=True), self.epsilon))
        # # loss_ac=torch.tensor(0)
        #----------------------------------------------------------------------------------------------
        # logits = self.gin(self.g,h)
        # print(logits.shape)
        logits = logits.unsqueeze(dim=1)
        logits = logits.mean(1)
        # print(logits.shape)
        # This is an equivalent replacement for tf.l2_normalize, see https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/math/l2_normalize for more information.
        # logits = logits / (torch.max(torch.norm(logits, dim=1, keepdim=True), self.epsilon))
        # loss_ac=torch.tensor(0)
        return logits,loss_ac

"""Torch Module for Graph Isomorphism Network layer"""
# pylint: disable= no-member, arguments-differ, invalid-name





class myGINConv(nn.Module):
    r"""Graph Isomorphism Network layer from `How Powerful are Graph
    Neural Networks? <https://arxiv.org/pdf/1810.00826.pdf>`__

    .. math::
        h_i^{(l+1)} = f_\Theta \left((1 + \epsilon) h_i^{l} +
        \mathrm{aggregate}\left(\left\{h_j^{l}, j\in\mathcal{N}(i)
        \right\}\right)\right)

    If a weight tensor on each edge is provided, the weighted graph convolution is defined as:

    .. math::
        h_i^{(l+1)} = f_\Theta \left((1 + \epsilon) h_i^{l} +
        \mathrm{aggregate}\left(\left\{e_{ji} h_j^{l}, j\in\mathcal{N}(i)
        \right\}\right)\right)

    where :math:`e_{ji}` is the weight on the edge from node :math:`j` to node :math:`i`.
    Please make sure that `e_{ji}` is broadcastable with `h_j^{l}`.

    Parameters
    ----------
    apply_func : callable activation function/layer or None
        If not None, apply this function to the updated node feature,
        the :math:`f_\Theta` in the formula, default: None.
    aggregator_type : str
        Aggregator type to use (``sum``, ``max`` or ``mean``), default: 'sum'.
    init_eps : float, optional
        Initial :math:`\epsilon` value, default: ``0``.
    learn_eps : bool, optional
        If True, :math:`\epsilon` will be a learnable parameter. Default: ``False``.
    activation : callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.

    Examples
    --------
    >>> import dgl
    >>> import numpy as np
    >>> import torch as th
    >>> from dgl.nn import GINConv
    >>>
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> feat = th.ones(6, 10)
    >>> lin = th.nn.Linear(10, 10)
    >>> conv = GINConv(lin, 'max')
    >>> res = conv(g, feat)
    >>> res
    tensor([[-0.4821,  0.0207, -0.7665,  0.5721, -0.4682, -0.2134, -0.5236,  1.2855,
            0.8843, -0.8764],
            [-0.4821,  0.0207, -0.7665,  0.5721, -0.4682, -0.2134, -0.5236,  1.2855,
            0.8843, -0.8764],
            [-0.4821,  0.0207, -0.7665,  0.5721, -0.4682, -0.2134, -0.5236,  1.2855,
            0.8843, -0.8764],
            [-0.4821,  0.0207, -0.7665,  0.5721, -0.4682, -0.2134, -0.5236,  1.2855,
            0.8843, -0.8764],
            [-0.4821,  0.0207, -0.7665,  0.5721, -0.4682, -0.2134, -0.5236,  1.2855,
            0.8843, -0.8764],
            [-0.1804,  0.0758, -0.5159,  0.3569, -0.1408, -0.1395, -0.2387,  0.7773,
            0.5266, -0.4465]], grad_fn=<AddmmBackward>)

    >>> # With activation
    >>> from torch.nn.functional import relu
    >>> conv = GINConv(lin, 'max', activation=relu)
    >>> res = conv(g, feat)
    >>> res
    tensor([[5.0118, 0.0000, 0.0000, 3.9091, 1.3371, 0.0000, 0.0000, 0.0000, 0.0000,
             0.0000],
            [5.0118, 0.0000, 0.0000, 3.9091, 1.3371, 0.0000, 0.0000, 0.0000, 0.0000,
             0.0000],
            [5.0118, 0.0000, 0.0000, 3.9091, 1.3371, 0.0000, 0.0000, 0.0000, 0.0000,
             0.0000],
            [5.0118, 0.0000, 0.0000, 3.9091, 1.3371, 0.0000, 0.0000, 0.0000, 0.0000,
             0.0000],
            [5.0118, 0.0000, 0.0000, 3.9091, 1.3371, 0.0000, 0.0000, 0.0000, 0.0000,
             0.0000],
            [2.5011, 0.0000, 0.0089, 2.0541, 0.8262, 0.0000, 0.0000, 0.1371, 0.0000,
             0.0000]], grad_fn=<ReluBackward0>)
    """

    def __init__(
        self,
        apply_func=None,
        aggregator_type="sum",
        init_eps=0,
        learn_eps=False,
        activation=None,
    ):
        super(myGINConv, self).__init__()
        self.apply_func = apply_func
        self._aggregator_type = aggregator_type
        self.activation = activation
        if aggregator_type not in ("sum", "max", "mean"):
            raise KeyError(
                "Aggregator type {} not recognized.".format(aggregator_type)
            )
        # to specify whether eps is trainable or not.
        if learn_eps:
            self.eps = th.nn.Parameter(th.FloatTensor([init_eps]))
        else:
            self.register_buffer("eps", th.FloatTensor([init_eps]))



    def forward(self, graph, feat, edge_weight=None):
        r"""

        Description
        -----------
        Compute Graph Isomorphism Network layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in})` and :math:`(N_{out}, D_{in})`.
            If ``apply_func`` is not None, :math:`D_{in}` should
            fit the input dimensionality requirement of ``apply_func``.
        edge_weight : torch.Tensor, optional
            Optional tensor on the edge. If given, the convolution will weight
            with regard to the message.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where
            :math:`D_{out}` is the output dimensionality of ``apply_func``.
            If ``apply_func`` is None, :math:`D_{out}` should be the same
            as input dimensionality.
        """
        _reducer = getattr(fn, self._aggregator_type)
        with graph.local_scope():
            aggregate_fn = fn.copy_u("h", "m")
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.num_edges()
                graph.edata["_edge_weight"] = edge_weight
                aggregate_fn = fn.u_mul_e("h", "_edge_weight", "m")

            feat_src, feat_dst = expand_as_pair(feat, graph)
            graph.srcdata["h"] = feat_src
            graph.update_all(aggregate_fn, _reducer("m", "neigh"))
            rst = (1 + self.eps) * feat_dst + graph.dstdata["neigh"]
            if self.apply_func is not None:
                rst = self.apply_func(rst)
            # activation
            # if self.activation is not None:
            #     rst = self.activation(rst)
            rst = torch.sigmoid(rst)
            return rst

