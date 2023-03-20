import torch
import torch.nn as nn

from models.attn import FullAttention, ProbAttention, AttentionLayer
from models.embed import DataEmbedding
from models.layers import LinearAE, EncoderLayer, autoencoder, LinearVAE, LinearAEn

from models.GCN import graph_constructor_agcn, prop

"""
输入数据是B*T*K*f，输出数据是重构出的B*T*K*f和未来一个时刻的预测值B*K*f
f = 1
"""

class Aformer(nn.Module):
    def __init__(self,
        enc_in, c_out, 
        # transformer
        d_model=512, n_heads=8, d_ff=512, e_layers=3,  
        embed='fixed', freq='h', dropout=0.0, attn='full', factor=5, 
        start_len=5,
        # factor probAttention中才会用到 
        activation='gelu', 
        output_attention = False,
        k=0, 
        use_gcn = False,
        # gcn才回用到
        num_nodes=None, subgraph_size=10, node_dim=10, tanhalpha = 3, static_feat=None, AE=None, seq_len=5
    ):
        super(Aformer, self).__init__()
        self.attn = attn
        self.AE = AE
        self.padding=0
        self.start_len=start_len

        enc_in = 1
        c_out = 1
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        Attn = ProbAttention if self.attn=='prob' else FullAttention
        if AE is None:
            AE_Module = None
        else:
            if AE == "VAE":
                AE_Module = LinearVAE(d_model, int(d_model/4)) 
            elif AE == "AEn":
                AE_Module = LinearAEn(num_nodes, int(num_nodes/2)) 
            else:
                AE_Module = LinearAE(d_model, int(d_model/4))
        self.autoencoder = autoencoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                    use_gcn=use_gcn,
                    subgraph_size=subgraph_size,
                    seq_len=seq_len
                ) for l in range(e_layers)
            ],
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                EncoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                    use_gcn=False
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
            aeLinear=AE_Module,
            # aeLinear=None,
            k=k
        )
        self.projection = nn.Linear(d_model, c_out, bias=True)
        self.recon_projection = nn.Linear(d_model, c_out, bias=True)

        # add gcn
        self.use_gcn = use_gcn
        self.num_nodes = num_nodes
        if use_gcn:
            # self.gc = graph_constructor2(num_nodes, subgraph_size, node_dim)
            # self.gc = graph_constructor3(num_nodes, subgraph_size, node_dim, alpha=tanhalpha)
            self.gc = graph_constructor_agcn(num_nodes, node_dim)


    def forward(self, x_enc, x_mark_enc=None, enc_self_mask=None, dec_self_mask=None):
        
        if self.use_gcn:
            adp = self.gc(torch.arange(self.num_nodes).to(x_enc.device)) # adp: n*n
        else:
            adp = None

        B, T, N = x_enc.shape
        if self.padding==0:
            dec_inp = torch.zeros([x_enc.shape[0], 1, x_enc.shape[-1]]).float().to(x_enc.device)
        elif self.padding==1:
            dec_inp = torch.ones([x_enc.shape[0], 1, x_enc.shape[-1]]).float().to(x_enc.device)
        enc_x = torch.cat([x_enc, dec_inp], dim=-2).float() # b*(t+1)*n
        dec_x = x_enc[:,-self.start_len:,:]                 # b*t*n
        recon_dec_x = torch.cat([dec_inp, x_enc], dim=-2).float() # b*(1+t)*n
        # recon_dec_x = torch.cat([dec_inp, x_enc[:,:-1,:]], dim=-2).float()
        # 添加位置编码
        B, T, N = enc_x.shape
        enc_x = enc_x.transpose(1, 2).reshape(B*N, T, 1) # [B, T, N] -> [B*N, T, 1]
        enc_out = self.enc_embedding(enc_x, x_mark_enc)
        enc_out = enc_out.reshape(B, N, T, -1) 

        B, T, N = dec_x.shape
        dec_x = dec_x.transpose(1, 2).reshape(B*N, T, 1) # [B, T, N] -> [B*N, T, 1]
        dec_x = self.enc_embedding(dec_x, None)
        dec_x = dec_x.reshape(B, N, T, -1) 

        B, T, N = recon_dec_x.shape
        recon_dec_x = recon_dec_x.transpose(1, 2).reshape(B*N, T, 1) # [B, T, N] -> [B*N, T, 1]
        recon_dec_x = self.enc_embedding(recon_dec_x, None)
        recon_dec_x = recon_dec_x.reshape(B, N, T, -1) 
        
        dec_out, recon_out, attns, embedding_sets = self.autoencoder(enc_out, dec_x, recon_dec_x, enc_self_mask=enc_self_mask, dec_self_mask=dec_self_mask, adp=adp) # [B, N, T+1, c]
        dec_out = self.projection(dec_out).squeeze() # [B, N, 1, 1]
        # dec_out = dec_out.transpose(1, 2) # [B, 1, N]

        recon_out = self.recon_projection(recon_out).squeeze()
        recon_out = recon_out.transpose(1, 2)

        mu, logvar = None, None
        if self.AE == 'VAE':
            mus, logvars = [], []
            for embeddings in embedding_sets:
                mus.append(embeddings[1])
                logvars.append(embeddings[2])
            mu = torch.cat(mus, dim=-1)
            logvar = torch.cat(logvars, dim=-1)

        return dec_out, recon_out, mu, logvar
