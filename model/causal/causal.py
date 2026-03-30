# coding: utf-8
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer, TransformerEncoder
import math

class causal(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.device = args.device
        self.loss = "WeightedProbBinaryCELoss"
        
        nloc = args.num_pois
        self.num_pois = args.num_pois
        ntime = args.num_times
        self.num_users = args.num_users
        self.num_regions = args.num_regions

        # from config
        user_dim = args.user_dim
        loc_dim = args.loc_dim
        time_dim = args.time_dim
        reg_dim = args.region_dim
        nhead_enc = args.nhead
        nlayers = args.nlayers
        dropout = args.dropout
        
        ninp = loc_dim + user_dim + time_dim + reg_dim
        
        self.emb_loc = Embedding(nloc, loc_dim, zeros_pad=True, scale=True)
        self.emb_reg = Embedding(self.num_regions, reg_dim, zeros_pad=True, scale=True)
        self.emb_time = Embedding(ntime+1, time_dim, zeros_pad=True, scale=True)
        self.emb_user = Embedding(self.num_users, user_dim, zeros_pad=True, scale=True)
        
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        
        self.enc_layer = TransformerEncoderLayer(ninp, nhead_enc, ninp, dropout, batch_first=True)
        self.encoder = TransformerEncoder(self.enc_layer, nlayers)

        self.region_pos_encoder = PositionalEmbedding(reg_dim, dropout, max_len=20)
        self.region_enc_layer = TransformerEncoderLayer(reg_dim, 1, reg_dim, dropout=dropout, batch_first=True)
        self.region_encoder = TransformerEncoder(self.region_enc_layer, 2)
        
        self.layer_norm = nn.LayerNorm(ninp)
        self.dropout = dropout
        
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)
        self.decoder = nn.Linear(ninp, args.num_pois)

    @staticmethod
    def _generate_square_mask_(sz, device):
        mask = (torch.triu(torch.ones(sz, sz).to(device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def calculate_loss(self, batch_data):

        user = batch_data['user_id'].to(self.device)
        loc = batch_data['POI_id'].to(self.device)
        time = batch_data['time_id'].to(self.device)
        region = batch_data['region_id'].to(self.device)
        ds = batch_data['mask'].to(self.device)

        user_id = user.unsqueeze(dim=1).expand(loc.shape[0], loc.shape[1])

        indices = torch.arange(user_id.shape[1], device=self.device).unsqueeze(0).expand(user_id.shape[0], user_id.shape[1])
        src_mask = (indices >= ds.unsqueeze(1)).to(torch.bool)
        att_mask = self._generate_square_mask_(loc.shape[1], self.device)

        B, L = loc.shape
        y_poi = torch.full((B, L), 0, dtype=torch.long, device=self.device)

        for i in range(B):
            end = ds[i].item()
            y_poi[i, :end] = torch.cat(
                (
                    batch_data['POI_id'][i, 1:end].to(self.device),
                    batch_data['y_POI_id']['POI_id'][i].unsqueeze(dim=-1).to(self.device)
                ),
                dim=-1
            )

        logits= self.forward(
            loc, region, user_id, time,
            att_mask, src_mask
        )
        
        loss_ce = self.loss_fn(logits.transpose(1, 2), y_poi)

        return loss_ce
        
    def forward(
        self,
        src_loc,
        src_reg,
        src_user,
        src_time,
        src_square_mask,
        src_binary_mask,
    ):
        loc_emb_src = self.emb_loc(src_loc)          # (B, L, d)
        user_emb_src = self.emb_user(src_user)       # (B, L, d)
        time_emb_src = self.emb_time(src_time)       # (B, L, d)

        reg_emb = self.emb_reg(src_reg)

        # concat: loc + reg + user + time
        src = torch.cat([loc_emb_src, reg_emb, user_emb_src, time_emb_src], dim=-1)  # (B, L, ninp)
        src = src * math.sqrt(src.size(-1))

        src = self.pos_encoder(src)  # (B, L, ninp)
        src = self.encoder(src, mask=src_square_mask, src_key_padding_mask=src_binary_mask)  # (B, L, ninp)

        output = self.decoder(src)
        return output

    def predict(self, batch_data):
        """
        只做 factual 推理：返回每个样本 mask 指定位置的 logits（与你原先一致）
        """
        user = batch_data['user_id'].to(self.device)
        loc = batch_data['POI_id'].to(self.device)
        time = batch_data['time_id'].to(self.device)
        region = batch_data['region_id'].to(self.device)
        ds = batch_data['mask'].to(self.device)

        user_id = user.unsqueeze(dim=1).expand(loc.shape[0], loc.shape[1])
        indices = torch.arange(user_id.shape[1], device=self.device).unsqueeze(0).expand(user_id.shape[0], user_id.shape[1])
        src_mask = (indices >= ds.unsqueeze(1)).to(torch.bool)
        att_mask = self._generate_square_mask_(loc.shape[1], self.device)

        # (B, L, num_pois)
        logits = self.forward(
            loc, region, user_id, time,
            att_mask, src_mask
        )
        B = user_id.shape[0]
        batch_indices = torch.arange(B, device=self.device)
        result = logits[batch_indices, ds - 1]   # (B, num_pois)
        return result

class Embedding(nn.Module):
    def __init__(self, vocab_size, num_units, zeros_pad=True, scale=True):
        '''Embeds a given Variable.
        Args:
          vocab_size: An int. Vocabulary size.
          num_units: An int. Number of embedding hidden units.
          zero_pad: A boolean. If True, all the values of the fist row (id 0)
            should be constant zeros.
          scale: A boolean. If True. the outputs is multiplied by sqrt num_units.
        '''
        super(Embedding, self).__init__()
        self.vocab_size = vocab_size
        self.num_units = num_units
        self.zeros_pad = zeros_pad
        self.scale = scale
        self.lookup_table = nn.Parameter(torch.Tensor(vocab_size, num_units))
        nn.init.xavier_normal_(self.lookup_table.data)
        if self.zeros_pad:
            self.lookup_table.data[0, :].fill_(0)

    def forward(self, inputs):
        if self.zeros_pad:
            self.padding_idx = 0
        else:
            self.padding_idx = -1
        outputs = F.embedding(
            inputs, self.lookup_table,
            self.padding_idx, None, 2, False, False)  # copied from torch.nn.modules.sparse.py

        if self.scale:
            outputs = outputs * (self.num_units ** 0.5)

        return outputs


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
    
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=120):
        super(PositionalEmbedding, self).__init__()
        self.pos_emb_table = Embedding(max_len, d_model, zeros_pad=False, scale=False)
        pos_vector = torch.arange(max_len)
        self.dropout = nn.Dropout(p=dropout)
        self.register_buffer('pos_vector', pos_vector)

    def forward(self, x):
        seq_len = x.size(1)
        pos_vector = self.pos_vector[:seq_len].unsqueeze(0).unsqueeze(-1)  # (1, seq_len, 1)
        pos_emb = self.pos_emb_table(pos_vector).squeeze(2)  # (1, seq_len, d_model)
        # 调整位置编码的形状以匹配输入 x (batch_size*seq_len, LEN_QUADKEY, d_model)
        pos_emb = pos_emb.unsqueeze(-2).repeat(x.size(0), 1, 1, 1).reshape(x.size(0), x.size(1), -1)
        x += pos_emb
        return self.dropout(x)