# coding: utf-8
from __future__ import print_function
from __future__ import division

from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from torchvision import models
from os.path import join, abspath, dirname
import math
from tqdm import tqdm
import numpy as np
from datetime import datetime

class TPG(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.device = args.device
        self.num_neg = 100
        self.temperature = 1
        self.loss = "WeightedProbBinaryCELoss"
        self.pure_time_prompt = True
        
        # from dataset
        # from train_dataset!!
        nloc = args.num_pois
        self.num_pois = args.num_pois
        ntime = args.num_time
        nquadkey = args.nquadkey
        self.region_id_map = args.region_id_map
        self.global_quadkeys = args.QUADKEY
        self.num_users = args.num_users

        # from config
        user_dim = 50
        loc_dim = 50
        time_dim = 50
        reg_dim = 50
        nhead_enc = 1
        nlayers = 2
        dropout = 0.5
        self.length = 9
        self.use_time_query = True
        self.use_time_loss = False
        self.loss_embedding_fusion = "multiply"
        
        ninp = loc_dim + user_dim + time_dim + reg_dim
        
        # essential
        self.matching_strategy = "divide"
        time_trg_dim = ninp
                 
        self.clip = False
        
        self.emb_loc = Embedding(nloc, loc_dim, zeros_pad=True, scale=True)
        self.emb_time_trg = Embedding(ntime+1, time_trg_dim, zeros_pad=True, scale=True)
        self.emb_reg = Embedding(nquadkey, reg_dim, zeros_pad=True, scale=True)
        self.emb_time = Embedding(ntime+1, time_dim, zeros_pad=True, scale=True)
        self.emb_user = Embedding(self.num_users, user_dim, zeros_pad=True, scale=True)
        
        if not ((user_dim == loc_dim) and (user_dim == time_dim) and (user_dim == reg_dim)):
            raise Exception("user, location, time and region should have the same embedding size!")
        
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        
        self.enc_layer = TransformerEncoderLayer(ninp, nhead_enc, ninp, dropout, batch_first=True)
        self.encoder = TransformerEncoder(self.enc_layer, nlayers)

        self.region_pos_encoder = PositionalEmbedding(reg_dim, dropout, max_len=20)
        self.region_enc_layer = TransformerEncoderLayer(reg_dim, 1, reg_dim, dropout=dropout, batch_first=True)
        self.region_encoder = TransformerEncoder(self.region_enc_layer, 2)

        self.lin = nn.Linear(loc_dim + reg_dim, ninp)
        self.time_lin = nn.Linear(time_trg_dim, ninp)
        
        ident_mat = torch.eye(ninp)
        self.register_buffer('ident_mat', ident_mat)
        
        self.layer_norm = nn.LayerNorm(ninp)
        self.dropout = dropout
        
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)
        self.decoder = nn.Linear(ninp, args.num_pois)
            
    def predict(self, batch_data):
        """
        Args:
            batch (Batch): a batch of input

        Returns:
            torch.tensor: predict result of this batch
        """
        # TODO 
        user, loc, time, region, ds = batch_data['user_id'].to(self.device), batch_data['POI_id'].to(self.device), batch_data['time_id'].to(self.device), \
                                                                              batch_data['region_id'].to(self.device), batch_data['mask'].to(self.device)
        
        user_id = user.unsqueeze(dim=1).expand(loc.shape[0], loc.shape[1])
        indices = torch.arange(user_id.shape[1]).unsqueeze(0).expand(user_id.shape[0], user_id.shape[1]).to(self.device)
        src_mask = (indices >= ds.unsqueeze(1)).to(torch.bool)
        att_mask = self._generate_square_mask_(loc.shape[1], self.device)
        
        B, L = loc.shape
        time_query = torch.full((B, L), 0, dtype=torch.long, device=self.device)
        for i in range(B):
            end = ds[i].item()
            time_query[i, :end] = torch.cat((batch_data['time_id'][i, 1:end].to(self.device), \
                batch_data['y_POI_id']['time_id'][i].unsqueeze(dim=-1).to(self.device)), dim=-1)
        
        # batch_size, seq_len, num_pois
        output = self.forward(loc, region, user_id, time, att_mask, src_mask, att_mask, time_query=time_query, is_training=False)
        
        batch_indices = torch.arange(batch_data['mask'].shape[0])
        result = output[batch_indices, batch_data['mask'] - 1]  
        return result

    @staticmethod
    def _generate_square_mask_(sz, device):
        mask = (torch.triu(torch.ones(sz, sz).to(device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def calculate_loss(self, batch_data):
        """
        Args:
            batch (Batch): a batch of input

        Returns:
            torch.tensor: return training loss
        """
        user, loc, time, region, ds = batch_data['user_id'].to(self.device), batch_data['POI_id'].to(self.device), batch_data['time_id'].to(self.device), \
                                                                              batch_data['region_id'].to(self.device), batch_data['mask'].to(self.device)
        
        user_id = user.unsqueeze(dim=1).expand(loc.shape[0], loc.shape[1])
        indices = torch.arange(user_id.shape[1]).unsqueeze(0).expand(user_id.shape[0], user_id.shape[1]).to(self.device)
        src_mask = (indices >= ds.unsqueeze(1)).to(torch.bool)
        att_mask = self._generate_square_mask_(loc.shape[1], self.device)
        
        B, L = loc.shape
        time_query = torch.full((B, L), 0, dtype=torch.long, device=self.device)
        y_poi = torch.full((B, L), 0, dtype=torch.long, device=self.device)
        for i in range(B):
            end = ds[i].item()
            time_query[i, :end] = torch.cat((batch_data['time_id'][i, 1:end].to(self.device), \
                batch_data['y_POI_id']['time_id'][i].unsqueeze(dim=-1).to(self.device)), dim=-1)
            y_poi[i, :end] = torch.cat((batch_data['POI_id'][i, 1:end].to(self.device), \
                batch_data['y_POI_id']['POI_id'][i].unsqueeze(dim=-1).to(self.device)), dim=-1)
            
        # batch_size, seq_len, num_pois
        output = self.forward(loc, region, user_id, time, att_mask, src_mask,
                                   att_mask, time_query=time_query)
        
        loss = self.loss_fn(output.transpose(1, 2), y_poi)

        return loss
    
    def forward(self, src_loc, src_reg, src_user, src_time, src_square_mask, src_binary_mask, mem_mask, time_query=None, is_training=True):
        # batch_size, seq_len
        loc_emb_src = self.emb_loc(src_loc)
        user_emb_src = self.emb_user(src_user)
        time_emb_src = self.emb_time(src_time)
        
        #  batch_size, seq_len, length, LEN_QUADKEY -> batch_size, seq_len, length, LEN_QUADKEY, reg_dim -> batch_size, seq_len, LEN_QUADKEY, reg_dim
        reg_emb = torch.mean(self.emb_reg(src_reg), dim=2)
        # batch_size * seq_len, LEN_QUADKEY, reg_dim
        reg_emb = reg_emb.reshape(-1, src_reg.shape[-1], reg_emb.shape[-1])
        
        reg_emb = self.region_pos_encoder(reg_emb)
        # batch_size * seq_len, LEN_QUADKEY, reg_dim
        reg_emb = self.region_encoder(reg_emb)
        # batch_size * seq_len, reg_dim -> batch_size, seq_len, reg_dim
        reg_emb = torch.mean(reg_emb, dim=-2).reshape(src_loc.shape[0], src_loc.shape[1], -1)

        src = torch.cat([loc_emb_src, reg_emb, user_emb_src, time_emb_src], dim=-1)
        src = src * math.sqrt(src.size(-1))
        
        # batch_size, seq_len, nnip
        src = self.pos_encoder(src)
        src = self.encoder(src, mask=src_square_mask, src_key_padding_mask=src_binary_mask)

        # batch_size, seq_len, nnip
        time_emb_trg = self.emb_time_trg(time_query)
        
        src = src.transpose(0, 1)
        time_emb_trg = time_emb_trg.transpose(0, 1)
        # multi-head attention    
        # seq_len, batch_size, nnip
        output, _ = F.multi_head_attention_forward(
            query=time_emb_trg,
            key=src,
            value=src,
            num_heads=1,
            embed_dim_to_check=src.shape[-1],
            in_proj_weight=None,
            in_proj_bias=None,
            bias_k=None,
            bias_v=None,
            add_zero_attn=None,
            dropout_p=0.0,
            out_proj_weight=self.ident_mat,
            out_proj_bias=None,
            training=is_training,
            key_padding_mask=src_binary_mask,
            need_weights=False,
            attn_mask=mem_mask,
            use_separate_proj_weight=True,
            q_proj_weight=self.ident_mat,
            k_proj_weight=self.ident_mat,
            v_proj_weight=self.ident_mat,
        )
        output += src
        output = output.transpose(0, 1)

        # batch_size, seq_len, nnip
        output = self.layer_norm(output)
        output = self.decoder(output)
        return output

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