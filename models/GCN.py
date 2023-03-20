
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

# embedding到邻接矩阵
class graph_constructor_agcn(nn.Module):
    def __init__(self, nnodes, dim, static_feat=None):
        super(graph_constructor_agcn, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)

        # self.device = device
        self.dim = dim
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)   
        else:
            nodevec1 = self.static_feat[idx,:]
        adj = torch.mm(nodevec1, nodevec1.transpose(1,0))
        return adj
   

class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self, x, A):
        x = torch.einsum('bctn,bmn->bctm',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out,bias=True):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=bias)

    def forward(self,x):
        return self.mlp(x)
# 图卷积
class prop(nn.Module):
    def __init__(self,c_in,c_out,gdep=1,dropout=0.3,alpha=0.05, subgraph_size=10, seq_len=5):
        super(prop, self).__init__()
        self.nconv = nconv()
        self.mlp = linear(c_in,c_out)
        self.mlp2 = linear(c_in,c_out)

        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha
        self.subgraph_size = subgraph_size
        seq_len=seq_len+1
        self.a = nn.Parameter(torch.empty((seq_len*2*c_out, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self,x,adj):
        # x.shape = b, c, t, n
        adj =  F.relu(adj)
        # adj = torch.softmax(adj, dim=1)

        adj = adj.to(x.device)
        d = adj.sum(1)
        dv = d
        adj = adj / dv.view(-1, 1)
        # print(adj)
        mask = torch.zeros(adj.size(0), adj.size(0)).to(adj.device)
        mask.fill_(float('0'))
        if self.training:
            s1,t1 = (adj + torch.rand_like(adj)*0.01).topk(self.subgraph_size,1)
        else:
            s1,t1 = (adj).topk(self.subgraph_size,1)
        mask.scatter_(1,t1,s1.fill_(1))
        adj1 = adj*mask # 取topk
        # mask = adj 
        
        Wx = self.mlp2(x) # b, c, t, n
        B,_,T,N = Wx.shape
        a_input = self._make_attention_input(Wx.transpose(-1, 1).reshape(B,N, -1)) 
        e = self.leakyrelu(torch.matmul(a_input, self.a)).squeeze(3)
        attention = torch.softmax(e, dim=2) # b,n,n

        h = x
        for i in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*self.nconv(h,mask*attention+adj1)
            # h = self.alpha*x + (1-self.alpha)*self.nconv(h,attention)

        ho = self.mlp(h)
        # ho = h
        return ho
    def _make_attention_input(self, v):
        K = v.size(1)
        blocks_repeating = v.repeat_interleave(K, dim=1)  # Left-side of the matrix
        blocks_alternating = v.repeat(1, K, 1)  # Right-side of the matrix
        combined = torch.cat((blocks_repeating, blocks_alternating), dim=2)  # (b, K*K, 2*window_size)
        return combined.view(v.size(0), K, K, -1)
