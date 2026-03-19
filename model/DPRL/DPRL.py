import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from enum import Enum
from math import pi
from .DPRL_utils import rotate_batch
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class Rnn(Enum):
    ''' The available RNN units '''
    
    RNN = 0
    GRU = 1    
    LSTM = 2    
    
    @staticmethod
    def from_string(name):
        if name == 'rnn':
            return Rnn.RNN
        if name == 'gru':
            return Rnn.GRU
        if name == 'lstm':
            return Rnn.LSTM        
        raise ValueError('{} not supported in --rnn'.format(name))        

class RnnFactory():
    ''' Creates the desired RNN unit. '''
    
    def __init__(self, rnn_type_str):
        self.rnn_type = Rnn.from_string(rnn_type_str)
                
    def __str__(self):
        if self.rnn_type == Rnn.RNN:
            return 'Use pytorch RNN implementation.'
        if self.rnn_type == Rnn.GRU:
            return 'Use pytorch GRU implementation.'
        if self.rnn_type == Rnn.LSTM:
            return 'Use pytorch LSTM implementation.'        
    
    def is_lstm(self):
        return self.rnn_type in [Rnn.LSTM]
        
    def create(self, hidden_size):
        if self.rnn_type == Rnn.RNN:
            return nn.RNN(hidden_size, hidden_size)
        if self.rnn_type == Rnn.GRU:
            return nn.GRU(hidden_size, hidden_size)
        if self.rnn_type == Rnn.LSTM:
            return nn.LSTM(hidden_size, hidden_size)

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=20):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe
        return self.dropout(x)
        
class User_Week_Distribution(nn.Module):
    def __init__(self,stamp_num):
        super().__init__()
        self.stamp_num=stamp_num
        self.sigma=nn.Parameter(torch.ones(self.stamp_num).view(self.stamp_num,1))

    def forward(self,x):
        self.sigma.data=torch.abs(self.sigma.data)
        learned_weight=1/torch.sqrt(2*pi*(self.sigma**2))*torch.exp(-(x**2)/(2*(self.sigma**2)))
        sum=torch.sum(learned_weight,dim=1,keepdim=True)
        return learned_weight/sum

class TransformerModel(nn.Module):
    def __init__(self, embed_size, nhead, nhid, nlayers, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embed_size = embed_size
        self.pos_encoder = PositionalEncoding(embed_size, dropout)
        encoder_layers = TransformerEncoderLayer(embed_size, nhead, nhid, dropout)  
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

    def generate_square_subsequent_mask(self, sz, device):
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, src, src_mask=None):
        if src_mask is None:
            src_mask = self.generate_square_subsequent_mask(src.size(0), src.device)
        src = src * math.sqrt(self.embed_size)
        src = self.pos_encoder(src)
        src = self.transformer_encoder(src, src_mask)

        return src
    
class DPRL(nn.Module):
 
    
    def __init__(self, args):
        super().__init__()
        self.input_size = input_size = args.num_pois  
        self.user_count = user_count = args.num_users
        self.hidden_size = hidden_size = args.hidden_dim
        slot_count = args.num_time_slots
        self.region_count = region_count = args.num_regions
        
        self.loc_count = args.num_pois
        self.device = args.device
        rnn_factory = RnnFactory(args.rnn)
        
        self.f_t = lambda delta_t, user_len: ((torch.cos(delta_t*2*np.pi/86400) + 1) / 2)*torch.exp(-(delta_t/86400*args.lambda_t)) 
        self.f_s = lambda delta_s, user_len: torch.exp(-(delta_s*args.lambda_s)) 
        
        self.cl_decay_steps = args.cl_decay_steps
        self.dr_ratio = args.dropout

        self.lambda_loc = args.lambda_loc
        self.lambda_user = args.lambda_user
        self.lambda_r = args.lambda_r
        self.use_weight = args.use_weight
        self.use_graph_user = args.use_graph_user
        self.use_spatial_graph = args.use_spatial_graph

        self.poi_dropout = nn.Dropout(self.dr_ratio)  
        self.region_dropout = nn.Dropout(0.2)
        self.dropout = nn.Dropout(0.2)
        self.encoder = nn.Embedding(input_size, hidden_size)  
        self.time_encoder = nn.Embedding(slot_count, hidden_size//2)         
        self.region_encoder = nn.Embedding(region_count, hidden_size)  
        self.user_encoder = nn.Embedding(user_count, hidden_size)  
        self.up_encoder = nn.Embedding(user_count, hidden_size)  
        self.ur_encoder = nn.Embedding(user_count, hidden_size)  
        
        self.rnn_size = hidden_size * 2
        self.rnn = rnn_factory.create(self.rnn_size)  
        self.rnn_r = rnn_factory.create(self.rnn_size)
        self.fc_size = self.rnn_size + hidden_size * 2 + hidden_size // 2
        self.fc = nn.Linear(self.fc_size, input_size)
        self.fc_r = nn.Linear(self.rnn_size * 2, hidden_size)
        
        mu = 0.0
        sd = 1.0 / self.hidden_size
        mem = torch.randn(self.user_count, self.hidden_size * 2) * sd + mu
        self.register_buffer("memory", mem) 
                
        self.cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=0)
        
    def handle_sequence(self, batch_data):
        poi_id, time, time_delta, region = batch_data['POI_id'], batch_data['timestamps'], batch_data['time_slot'], batch_data['region']
        
        seq_len = batch_data['mask']
        
        y_poi_id, y_time, y_time_delta, y_region = batch_data['y_POI_id']['POI_id'], batch_data['y_POI_id']['timestamps'], batch_data['y_POI_id']['time_slot'], batch_data['y_POI_id']['region']
        
        B, L = batch_data['POI_id'].size()
        y_poi_seq = torch.full((B, L), 0, dtype=torch.long, device=self.device)
        y_time_seq = torch.full((B, L), 0, dtype=torch.long, device=self.device)
        y_time_delta_seq = torch.full((B, L), 0, dtype=torch.float, device=self.device)
        y_region_seq = torch.full((B, L), 0, dtype=torch.long, device=self.device)
        
        for i in range(B):
            end = seq_len[i].item()
            y_poi_seq[i, :end] = torch.cat((poi_id[i, 1:end], y_poi_id[i].unsqueeze(dim=-1)), dim=-1)
            y_time_seq[i, :end] = torch.cat((time[i, 1:end], y_time[i].unsqueeze(dim=-1)), dim=-1)
            y_time_delta_seq[i, :end] = torch.cat((time_delta[i, 1:end], y_time_delta[i].unsqueeze(dim=-1)), dim=-1)
            y_region_seq[i, :end] = torch.cat((region[i, 1:end], y_region[i].unsqueeze(dim=-1)), dim=-1)
        
        batch_data['y_POI_id']['POI_id'] = y_poi_seq
        batch_data['y_POI_id']['timestamps'] = y_time_seq
        batch_data['y_POI_id']['time_slot'] = y_time_delta_seq
        batch_data['y_POI_id']['region'] = y_region_seq
        
        return batch_data 

    def forward(self, batch_data):        
        coordinates = torch.stack([batch_data['latitude'], batch_data['longitude']], dim=-1)
        x, t, t_slot, r, s, y_t_slot, active_user = batch_data['POI_id'].transpose(0, 1), batch_data['timestamps'].transpose(0, 1), batch_data['time_slot'].transpose(0, 1), \
        batch_data['region'].transpose(0, 1), coordinates.transpose(0, 1), batch_data['y_POI_id']['time_slot'].long().transpose(0, 1), batch_data['user_id']
        
        h = self.memory[active_user].unsqueeze(dim=0)
        
        seq_len, batch_size = x.size()
        x_emb = self.encoder(x)
        rg_emb = self.region_encoder(r)
        t_emb = self.time_encoder(t_slot)
        t_emb_nx = self.time_encoder(y_t_slot)

        p_u = self.user_encoder(active_user).repeat(seq_len, 1, 1)  
        up_pref = self.up_encoder(active_user).repeat(seq_len, 1, 1)
        ur_pref = self.ur_encoder(active_user).repeat(seq_len, 1, 1)
        up_pref = rotate_batch(up_pref, t_emb, self.hidden_size)
        ur_pref = rotate_batch(ur_pref, t_emb, self.hidden_size)
        
        new_x_emb = torch.cat([x_emb, up_pref], dim=-1)  
        new_rg_emb = torch.cat([rg_emb, ur_pref], dim=-1)  
        out, h_ = self.rnn(new_x_emb, h)        
        out_r, h_r = self.rnn_r(new_rg_emb, h) 
        out = self.poi_dropout(out) 
        out_r = self.region_dropout(out_r) 
        
        user_loc_similarity = torch.exp(-(torch.norm(up_pref - x_emb, p=2, dim=-1))).to(x.device)
        user_region_similarity = torch.exp(-(torch.norm(ur_pref - rg_emb, p=2, dim=-1))).to(x.device)
        
        out_w = torch.zeros(seq_len, batch_size, self.rnn_size, device=x.device)
        out_wr = torch.zeros(seq_len, batch_size, self.rnn_size, device=x.device)
            
        for i in range(seq_len):  
            dist_t = torch.clamp(t[i].unsqueeze(0) - t[:i+1], min=0)
            dist_s = torch.norm(s[i].unsqueeze(0) - s[:i+1], dim=-1)
            a_j = self.f_t(dist_t, batch_size).unsqueeze(-1)
            b_j = self.f_s(dist_s, batch_size).unsqueeze(-1)
            # Compute the weights
            w_j = a_j * b_j
            w_jp = w_j * user_loc_similarity[:i+1].unsqueeze(-1) + 1e-10     
            w_jr = w_j * user_region_similarity[:i+1].unsqueeze(-1) + 1e-10  
            sum_wp = w_jp.sum(dim=0)
            sum_wr = w_jr.sum(dim=0)
            out_w[i] = (w_jp * out[:i+1]).sum(dim=0) / sum_wp
            out_wr[i] = (w_jr * out_r[:i+1]).sum(dim=0) / sum_wr
        
        out_pu_r = self.fc_r(torch.cat([out_wr, p_u, ur_pref], dim=-1))
        y_linear_r = out_pu_r.matmul(self.region_encoder.weight.transpose(1, 0)).transpose(0, 1)
        out_pu = torch.cat([out_w+out_wr, p_u, up_pref+ur_pref, t_emb_nx], dim=-1)
        out_pu = self.dropout(out_pu) 
        y_linear = self.fc(out_pu).transpose(0, 1)

        end = batch_data["end"]          # shape: (batch_size,), values 0/1 or bool
        end_mask = end.to(torch.bool)   # 关键：转成布尔mask
        with torch.no_grad():
            self.memory[active_user[end_mask]] = h.squeeze(0)[end_mask]

        return y_linear, y_linear_r, h
    
    def focal_loss(self, pred_reg, y_reg, alpha=0.25, gamma=2.0):
        """
        pred_reg: Tensor, shape (T*B, region_count), 模型预测的logits
        y_reg: Tensor, shape (T*B,), Groundtruth Region类别索引
        alpha: float, Focal Loss的权重参数
        gamma: float, Focal Loss的难样本关注参数
        """
        probs = torch.softmax(pred_reg, dim=-1)  
        pred_r_values, pred_r_indices = torch.topk(probs, k=10, dim=-1)  

        positive_probs = probs[range(len(y_reg)), y_reg]
        
        focal_loss = -alpha * (1 - positive_probs)**gamma * torch.log(positive_probs + 1e-8)  
        margin_loss = torch.relu(1 + pred_r_values - positive_probs.unsqueeze(-1))
        
        return focal_loss.mean(), margin_loss.mean()
    
    def calculate_loss(self, batch_data):
        batch_data = self.handle_sequence(batch_data)
        y, y_r = batch_data['y_POI_id']['POI_id'], batch_data['y_POI_id']['region']
        
        out, out_r, h = self.forward(batch_data)

        l = self.cross_entropy_loss(out.transpose(1, 2), y)
        l_r, _ = self.focal_loss(out_r.reshape(-1, self.region_count), y_r.reshape(-1))
        l = l + self.lambda_r * l_r
        
        return l
    
    def predict(self, batch_data):
        batch_data = self.handle_sequence(batch_data)
        y_pred_poi, _, _ = self.forward(batch_data)
        
        batch_indices = torch.arange(batch_data['mask'].shape[0])
        y_pred = y_pred_poi[batch_indices, batch_data['mask'] - 1] 
        
        return y_pred


def create_h0_strategy(hidden_size, is_lstm):
    if is_lstm:        
        return LstmStrategy(hidden_size, FixNoiseStrategy(hidden_size), FixNoiseStrategy(hidden_size))        
    else:        
        return FixNoiseStrategy(hidden_size)

class H0Strategy():
    
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size
    
    def on_init(self, user_len, device):
        pass
    
    def on_reset(self, user):
        pass
    
    def on_reset_test(self, user, device):
        return self.on_reset(user)


class FixNoiseStrategy(H0Strategy):
    ''' use fixed normal noise as initialization. '''
    
    def __init__(self, hidden_size):
        super().__init__(hidden_size)
        mu = 0
        sd = 1/self.hidden_size
        self.h0 = torch.randn(self.hidden_size, requires_grad=False) * sd + mu
    
    def on_init(self, user_len, device):
        hs = []
        for i in range(user_len):
            hs.append(self.h0)
        return torch.stack(hs, dim=0).view(1, user_len, self.hidden_size).to(device)
    
    def on_reset(self, user):
        return self.h0

class LstmStrategy(H0Strategy):
    ''' creates h0 and c0 using the inner strategy '''
    
    def __init__(self, hidden_size, h_strategy, c_strategy):
        super(LstmStrategy, self).__init__(hidden_size)
        self.h_strategy = h_strategy
        self.c_strategy = c_strategy
    
    def on_init(self, user_len, device):
        h = self.h_strategy.on_init(user_len, device)
        c = self.c_strategy.on_init(user_len, device)
        return (h,c)
    
    def on_reset(self, user):
        h = self.h_strategy.on_reset(user)
        c = self.c_strategy.on_reset(user)
        return (h,c)
