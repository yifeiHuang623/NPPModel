import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from enum import Enum
from math import pi

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

    
class GraphFlashback(nn.Module):
 
    
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


        rnn = RnnFactory(args.rnn)
        self.rnn = rnn.create(self.hidden_size)
        self.fc = nn.Linear(2*self.hidden_size, self.input_size)
        
        mu = 0.0
        sd = 1.0 / self.hidden_size
        mem = torch.randn(self.user_count, self.hidden_size) * sd + mu
        self.register_buffer("memory", mem) 
                
        self.cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=0)
        
    def handle_sequence(self, batch_data):
        poi_id, time = batch_data['POI_id'], batch_data['timestamps']
        
        seq_len = batch_data['mask']
        
        y_poi_id, y_time = batch_data['y_POI_id']['POI_id'], batch_data['y_POI_id']['timestamps']
        
        B, L = batch_data['POI_id'].size()
        y_poi_seq = torch.full((B, L), 0, dtype=torch.long, device=self.device)
        y_time_seq = torch.full((B, L), 0, dtype=torch.long, device=self.device)
        
        for i in range(B):
            end = seq_len[i].item()
            y_poi_seq[i, :end] = torch.cat((poi_id[i, 1:end], y_poi_id[i].unsqueeze(dim=-1)), dim=-1)
            y_time_seq[i, :end] = torch.cat((time[i, 1:end], y_time[i].unsqueeze(dim=-1)), dim=-1)
            
        batch_data['y_POI_id']['POI_id'] = y_poi_seq
        batch_data['y_POI_id']['timestamps'] = y_time_seq
        
        return batch_data 

    def forward(self, batch_data):        
        coordinates = torch.stack([batch_data['latitude'], batch_data['longitude']], dim=-1)
        
        x, t, s, active_user = batch_data['POI_id'].transpose(0, 1), batch_data['timestamps'].transpose(0, 1), coordinates.transpose(0, 1), batch_data['user_id']
        
        # batch_size, seq_len
        seq_len, user_len = x.size()
        h = self.memory[active_user].unsqueeze(dim=0)
        
        x_emb = self.encoder(x)
        # 是否用GCN来更新user embedding
        if self.use_graph_user:
            # I_f = identity(self.friend_graph.shape[0], format='coo')
            # friend_graph = (self.friend_graph * self.lambda_user + I_f).astype(np.float32)
            # friend_graph = calculate_random_walk_matrix(friend_graph)
            # friend_graph = sparse_matrix_to_tensor(friend_graph).to(x.device)
            friend_graph = self.friend_graph.to(x.device)
            # AX
            user_emb = self.user_encoder(torch.LongTensor(
                list(range(self.user_count))).to(x.device))
            user_encoder_weight = torch.sparse.mm(friend_graph, user_emb).to(
                x.device)  # (user_count, hidden_size)

            if self.use_weight:
                user_encoder_weight = self.user_gconv_weight(
                    user_encoder_weight)
            p_u = torch.index_select(
                user_encoder_weight, 0, active_user.squeeze())
        else:
            p_u = self.user_encoder(active_user)  # (1, user_len, hidden_size)
            # (user_len, hidden_size)
            p_u = p_u.view(user_len, self.hidden_size)

        p_u = self.user_encoder(active_user)  # (1, user_len, hidden_size)
        p_u = p_u.view(user_len, self.hidden_size)
        # AX,即GCN
        graph = self.graph.to(x.device)
        loc_emb = self.encoder(torch.LongTensor(
            list(range(self.input_size))).to(x.device))
        encoder_weight = torch.sparse.mm(graph, loc_emb).to(
            x.device)  # (input_size, hidden_size)
        
        if self.use_spatial_graph:
            spatial_graph = (self.spatial_graph *
                             self.lambda_loc + self.I).astype(np.float32)
            spatial_graph = calculate_random_walk_matrix(spatial_graph)
            spatial_graph = sparse_matrix_to_tensor(
                spatial_graph).to(x.device)  # sparse tensor gpu
            encoder_weight += torch.sparse.mm(spatial_graph,
                                              loc_emb).to(x.device)
            encoder_weight /= 2  # 求均值
       
        new_x_emb = []
        for i in range(seq_len):
            # (user_len, hidden_size)
            temp_x = torch.index_select(encoder_weight, 0, x[i])
            new_x_emb.append(temp_x)

        x_emb = torch.stack(new_x_emb, dim=0)  

        # user-poi
        loc_emb = self.encoder(torch.LongTensor(
            list(range(self.input_size))).to(x.device))
        encoder_weight = loc_emb
        interact_graph = self.interact_graph.to(x.device)
        encoder_weight_user = torch.sparse.mm(
            interact_graph, encoder_weight).to(x.device)

        user_preference = torch.index_select(
            encoder_weight_user, 0, active_user.squeeze()).unsqueeze(0)
        # print(user_preference.size())
        user_loc_similarity = torch.exp(
            -(torch.norm(user_preference - x_emb, p=2, dim=-1))).to(x.device)
        user_loc_similarity = user_loc_similarity.permute(1, 0)

        out, h = self.rnn(x_emb, h)  # (seq_len, user_len, hidden_size)
        out_w = torch.zeros(seq_len, user_len,
                            self.hidden_size, device=x.device)
        
        for i in range(seq_len):
            sum_w = torch.zeros(user_len, 1, device=x.device)  # (200, 1)
            for j in range(i + 1):
                dist_t = t[i] - t[j]
                dist_s = torch.norm(s[i] - s[j], dim=-1)
                a_j = self.f_t(dist_t, user_len)  # (user_len, )
                b_j = self.f_s(dist_s, user_len)
                a_j = a_j.unsqueeze(1)  # (user_len, 1)
                b_j = b_j.unsqueeze(1)
                w_j = a_j * b_j + 1e-10  # small epsilon to avoid 0 division
                w_j = w_j * user_loc_similarity[:, j].unsqueeze(1)  # (user_len, 1)
                sum_w += w_j
                out_w[i] += w_j * out[j]  # (user_len, hidden_size)
            out_w[i] /= sum_w
            
        out_pu = torch.zeros(seq_len, user_len, 2 *
                             self.hidden_size, device=x.device)
        for i in range(seq_len):
            # (user_len, hidden_size * 2)
            out_pu[i] = torch.cat([out_w[i], p_u], dim=1)

        y_linear = self.fc(out_pu)  # (seq_len, user_len, loc_count)
        
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
