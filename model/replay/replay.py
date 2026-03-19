import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from enum import Enum
from math import pi
from .replay_utils import generate_tensor_of_distribution

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
            return nn.RNN(hidden_size, hidden_size, batch_first=True)
        if self.rnn_type == Rnn.GRU:
            return nn.GRU(hidden_size, hidden_size, batch_first=True)
        if self.rnn_type == Rnn.LSTM:
            return nn.LSTM(hidden_size, hidden_size, batch_first=True)

        
        
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

    
class replay(nn.Module):
 
    
    def __init__(self, args):
        super().__init__()
        self.input_size = args.num_pois  
        self.user_count = args.num_users
        self.hidden_size = args.hidden_dim
        self.loc_count = args.num_pois
        self.device = args.device
        
        self.f_t = lambda delta_t, user_len: ((torch.cos(delta_t*2*np.pi/86400) + 1) / 2)*torch.exp(-(delta_t/86400*args.lambda_t)) 
        self.f_s = lambda delta_s, user_len: torch.exp(-(delta_s*args.lambda_s)) 
        self.week_matrix=generate_tensor_of_distribution(168).to(self.device)

        self.week_weight_index=torch.tensor([x-84 for x in range(168)]).repeat(168,1).to(args.device)
    
        self.encoder = nn.Embedding(self.input_size, self.hidden_size) # location embedding
        self.user_encoder = nn.Embedding(self.user_count, self.hidden_size) # user embedding
        self.week_encoder=nn.Embedding(24*7,self.hidden_size//2)


        rnn = RnnFactory(args.rnn)
        self.rnn = rnn.create(self.hidden_size)
        self.fc = nn.Linear(3*self.hidden_size-self.hidden_size//2, self.input_size) # create outputs in lenght of locations
        self.fcpt= nn.Linear(2*self.hidden_size-self.hidden_size//2, self.hidden_size)
        self.week_distribution=User_Week_Distribution(168)
        
        mu = 0.0
        sd = 1.0 / self.hidden_size
        mem = torch.randn(self.user_count, self.hidden_size) * sd + mu
        self.register_buffer("memory", mem) 
                
        self.cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=0)
        
    def handle_sequence(self, batch_data):
        poi_id, time, time_delta = batch_data['POI_id'], batch_data['timestamps'], batch_data['time_slot']
        
        seq_len = batch_data['mask']
        
        y_poi_id, y_time, y_time_delta = batch_data['y_POI_id']['POI_id'], batch_data['y_POI_id']['timestamps'], batch_data['y_POI_id']['time_slot']
        
        B, L = batch_data['POI_id'].size()
        y_poi_seq = torch.full((B, L), 0, dtype=torch.long, device=self.device)
        y_time_seq = torch.full((B, L), 0, dtype=torch.long, device=self.device)
        y_time_delta_seq = torch.full((B, L), 0, dtype=torch.float, device=self.device)
        
        for i in range(B):
            end = seq_len[i].item()
            y_poi_seq[i, :end] = torch.cat((poi_id[i, 1:end], y_poi_id[i].unsqueeze(dim=-1)), dim=-1)
            y_time_seq[i, :end] = torch.cat((time[i, 1:end], y_time[i].unsqueeze(dim=-1)), dim=-1)
            y_time_delta_seq[i, :end] = torch.cat((time_delta[i, 1:end], y_time_delta[i].unsqueeze(dim=-1)), dim=-1)
        
        batch_data['y_POI_id']['POI_id'] = y_poi_seq
        batch_data['y_POI_id']['timestamps'] = y_time_seq
        batch_data['y_POI_id']['time_slot'] = y_time_delta_seq
        
        return batch_data 

    def forward(self, batch_data):        
        coordinates = torch.stack([batch_data['latitude'], batch_data['longitude']], dim=-1)
        
        x, t, t_slot, s, y_t_slot, active_user, lengths = batch_data['POI_id'], batch_data['timestamps'], batch_data['time_slot'], coordinates, \
                                                batch_data['y_POI_id']['time_slot'].long(), batch_data['user_id'], batch_data['mask']
        
        # batch_size, seq_len
        batch_size, seq_len = x.size()
        # 1, batch_size, hidden_size
        h = self.memory[active_user].unsqueeze(dim=0)
        
        week_weight=self.week_distribution(self.week_weight_index).view(168,168)
        new_week_weight1=week_weight.index_select(0,t_slot.view(-1)).view(batch_size, seq_len,168,1)
        new_week_weight2=week_weight.index_select(0,y_t_slot.view(-1)).view(batch_size, seq_len,168,1)

        w_t1=self.week_matrix.index_select(0,t_slot.view(-1)).view(batch_size, seq_len,-1)
        w_t1=self.week_encoder(w_t1).permute(0,1,3,2)#seq*batch_size*5*168

        w_t1=torch.matmul(w_t1,new_week_weight1).squeeze(dim=-1)
        t_emb1 = w_t1
        w_t2=self.week_matrix.index_select(0,y_t_slot.view(-1)).view(batch_size, seq_len,-1)
        w_t2=self.week_encoder(w_t2).permute(0,1,3,2)#seq*batch_size*5*168
        w_t2=torch.matmul(w_t2,new_week_weight2).squeeze(dim=-1)
        t_emb2 = w_t2

        x_emb = self.encoder(x)        
        poi_time=self.fcpt(torch.cat((x_emb,t_emb1),dim=-1))
        out, h = self.rnn(poi_time, h)

        # seq_len, batch_size, hidden_size
        out = out.transpose(0, 1)
        
        T, U, H = out.shape

        device = out.device
        dtype = out.dtype
        t = t.transpose(0, 1).to(device=device, dtype=dtype)
        s = s.transpose(0, 1).to(device=device, dtype=dtype)
        t_emb2 = t_emb2.transpose(0,1).to(device=device, dtype=dtype)

        # dist_t/dist_s: (T,T,U)
        
        dist_t = torch.clamp(t[:, None, :] - t[None, :, :], min=0)                          # (T, T, U)
        dist_s = torch.norm(s[:, None, :, :] - s[None, :, :, :], dim=-1) # (T, T, U)

        # a,b: (T,T,U) —— f_t/f_s 必须支持这种输入
        a = self.f_t(dist_t, U).to(dtype)
        b = self.f_s(dist_s, U).to(dtype)

        w = a * b + 1e-10                                                # (T,T,U)

        # mask j<=i
        mask = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool))
        w = w.masked_fill(~mask[:, :, None], 0.0)                        # (T,T,U)

        # 加权聚合：对 j 求和
        # w: (T,T,U), out: (T,U,H)
        num = torch.einsum("tju,juh->tuh", w, out)                        # (T,U,H)
        den = w.sum(dim=1).unsqueeze(-1).clamp_min(1e-10)                # (T,U,1)
        out_w = num / den                                                 # (T,U,H)

        # user embedding: (U,H) -> broadcast to (T,U,H)
        p_u = self.user_encoder(active_user)                              # 通常 (U,H) 或 (B,H)
        p_u = p_u.reshape(U, H).to(device=device, dtype=dtype)            # (U,H)
        p_u_exp = p_u.unsqueeze(0).expand(T, U, H)                        # (T,U,H)

        out_pu = torch.cat([out_w, p_u_exp, t_emb2], dim=-1)              # (T,U,2H+E2)
        y_linear = self.fc(out_pu)                                        # (T,U, ...)

        # 如果你后面需要 (U,T,...)：
        y_linear = y_linear.transpose(0, 1)                               # (U,T,...)
        
        end = batch_data["end"]          # shape: (batch_size,), values 0/1 or bool
        end_mask = end.to(torch.bool)   # 关键：转成布尔mask
        with torch.no_grad():
                self.memory[active_user[end_mask]] = h.squeeze(0)[end_mask]
                
        return y_linear
    
    def calculate_loss(self, batch_data):
        batch_data = self.handle_sequence(batch_data)
        y = batch_data['y_POI_id']['POI_id']
        
        out = self.forward(batch_data).transpose(1, 2)
        l = self.cross_entropy_loss(out, y)
        return l
    
    def predict(self, batch_data):
        batch_data = self.handle_sequence(batch_data)
        y_pred_poi = self.forward(batch_data)
        
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
