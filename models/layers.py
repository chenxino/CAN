import torch
import torch.nn as nn
import torch.nn.functional as F
from models.GCN import prop
from torch.autograd import Variable
import copy

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu", use_gcn=False, gcn_depth=1, propalpha=0.05, subgraph_size=10, seq_len=5):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.attention = attention
        self.use_gcn = use_gcn
        self.conv1 = nn.Conv2d(in_channels=d_model, out_channels=d_ff, kernel_size=(1,1))
        # if use_gcn:
        #     self.conv1 = prop(d_model, d_ff, gdep = gcn_depth, dropout = dropout, alpha = propalpha)
        # else:
        #     self.conv1 = nn.Conv2d(in_channels=d_model, out_channels=d_ff, kernel_size=(1,1))
        if self.use_gcn:
            self.conv2 = prop(d_ff, d_model, gdep = gcn_depth, dropout = dropout, alpha = propalpha, subgraph_size=subgraph_size, seq_len=seq_len)
        else:
            self.conv2 = nn.Conv2d(in_channels=d_ff, out_channels=d_model, kernel_size=(1,1))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, adp=None):
        # x [B, N, L, D]
        # x = x + self.dropout(self.attention(
        #     x, x, x,
        #     attn_mask = attn_mask
        # ))
        new_x, attn = self.attention(
            x, x, x,
            attn_mask = attn_mask
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)

        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        if adp is None or self.use_gcn is False:
            y = self.dropout(self.conv2(y).transpose(-1,1))
        else:
            y = self.dropout(self.conv2(y,adp).transpose(-1,1))
        # y = self.dropout(self.conv2(y).transpose(-1,1))

        return self.norm2(x+y), attn

class LinearAE(nn.Module):
    def __init__(self, in_dim, hid_dim):
        super(LinearAE, self).__init__()
        self.encoder = nn.Linear(in_dim, hid_dim)
        self.decoder = nn.Linear(hid_dim, in_dim)
    def forward(self, x):
        embedding = self.encoder(x)
        dec_out = self.decoder(embedding)
        return dec_out, embedding

class LinearAEn(nn.Module):
    def __init__(self, in_dim, hid_dim):
        super(LinearAEn, self).__init__()
        self.encoder = nn.Linear(in_dim, hid_dim)
        self.decoder = nn.Linear(hid_dim, in_dim)
    def forward(self, x):
        x = x.transpose(1,-1)
        embedding = self.encoder(x)
        dec_out = self.decoder(embedding)
        dec_out = dec_out.transpose(1,-1)
        return dec_out, embedding

class LinearVAE(nn.Module):
    def __init__(self, in_dim, hid_dim):
        super(LinearVAE, self).__init__()
        self.encoder_mu = nn.Linear(in_dim, hid_dim)
        self.encoder_logvar = nn.Linear(in_dim, hid_dim)
        self.decoder = nn.Linear(hid_dim, in_dim)
    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_().to(mu.device)
        eps = Variable(eps)
        return eps.mul(std).add_(mu)
    def forward(self, x):
        mu = self.encoder_mu(x)
        logvar = self.encoder_logvar(x)
        embedding = self.reparametrize(mu, logvar)
        dec_out = self.decoder(embedding) # B,N,f
        return dec_out, (embedding, mu, logvar)

class autoencoder(nn.Module):
    def __init__(self, encoder_attn_layers, decoder_attn_layers, recoder_attn_layers, norm_layer=None, aeLinear =None, padding=0, k=0):
        super(autoencoder, self).__init__()
        self.encoder_layers = nn.ModuleList(encoder_attn_layers)
        self.decoder_layers = nn.ModuleList(decoder_attn_layers)
        self.recoder_layers = nn.ModuleList(recoder_attn_layers)
        self.norm = norm_layer
        self.aeLinear = aeLinear
        self.padding = padding
        self.k = k

    def forward(self, x_enc, x_dec, recon_x_dec, enc_self_mask=None, dec_self_mask=None, adp=None):
        # x_enc [B, N, T+1, f]
        # x_dec [B, N, T/2+2, f]
        attns = []
        embedding_sets = []
        for index, (encoder_layer, decoder_layer, recoder_layer) in enumerate(zip(self.encoder_layers, self.decoder_layers, self.recoder_layers)):
            x_enc, attn = encoder_layer(x_enc, attn_mask=enc_self_mask, adp=adp)

            if index >= self.k:
                if self.aeLinear is not None:
                    enc_out, embedding_set = self.aeLinear(x_enc[..., -1, :]) # B N f
                else:
                    enc_out, embedding_set = x_enc[..., -1, :], x_enc[..., -1, :]
                embedding_sets.append(embedding_set)
                # enc_out.unsqueeze(1)
                # print(enc_out.shape)
                x_dec = torch.cat([enc_out.unsqueeze(-2), x_dec], dim=-2).float()
                recon_x_dec = torch.cat([enc_out.unsqueeze(-2), recon_x_dec], dim=-2).float()
            x_dec, attn2 = decoder_layer(x_dec, attn_mask=dec_self_mask, adp=adp)
            
            recon_x_dec, attn3 = recoder_layer(recon_x_dec, attn_mask=dec_self_mask, adp=adp)

            attns.append(attn)

        if self.norm is not None:
            recon_x_dec = self.norm(recon_x_dec)
            x_dec = self.norm(x_dec)

        return x_dec[..., -1:, :], recon_x_dec[..., -x_enc.size(-2):-1,:], attns, embedding_sets
        # return x_dec[..., -1:, :], recon_x_dec[..., -x_enc.size(-2):-1,:], attns, embedding_sets
