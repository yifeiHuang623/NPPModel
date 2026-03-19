import torch
from torch import nn

import math

import torch
from torch import nn
from .MCLP_utils import MyEmbedding, PositionalEncoding
from .MCLP_utils  import TransEncoder, LSTMEncoder
from .MCLP_utils  import MyFullyConnect
from .MCLP_utils import UserNet


class MCLP(nn.Module):
    def __init__(self, args):
        super(MCLP, self).__init__()
        self.args = args
        self.base_dim = args.base_dim
        self.topic_num = args.topic_num

        self.embedding_layer = MyEmbedding(args)

        if args.encoder_type == 'trans':
            emb_dim = self.base_dim
            self.positional_encoding = PositionalEncoding(emb_dim=emb_dim)
            self.encoder = TransEncoder(args)
        if args.encoder_type == 'lstm':
            self.encoder = LSTMEncoder(args)

        fc_input_dim = self.base_dim + self.base_dim

        if args.at_type != 'none':
            self.at_net = ArrivalTime(args)
            fc_input_dim += self.base_dim

        if self.topic_num > 0:
            self.user_net = UserNet(input_dim=self.topic_num, output_dim=self.base_dim)
            fc_input_dim += self.base_dim

        self.fc_layer = MyFullyConnect(input_dim=fc_input_dim,
                                       output_dim=args.num_pois)
        self.out_dropout = nn.Dropout(0.1)
        
        self.loss_func = nn.CrossEntropyLoss(ignore_index=0)
        
        self.device = args.device

    def forward(self, batch_data):
        user_x = batch_data['user_id']
        loc_x = batch_data['POI_id']
        hour_x = batch_data['hour']
        if self.topic_num > 0:
            pre_embedded = batch_data['user_topic_loc']
        batch_size, sequence_length = loc_x.shape

        loc_embedded, timeslot_embedded, user_embedded = self.embedding_layer(batch_data)
        time_embedded = timeslot_embedded[hour_x]

        lt_embedded = loc_embedded + time_embedded

        if self.args.encoder_type == 'trans':
            future_mask = torch.triu(torch.ones(sequence_length, sequence_length), diagonal=1).to(lt_embedded.device)
            future_mask = future_mask.masked_fill(future_mask == 1, float('-inf')).bool()
            encoder_out = self.encoder(self.positional_encoding(lt_embedded * math.sqrt(self.base_dim)),
                                       src_mask=future_mask)
        if self.args.encoder_type == 'lstm':
            encoder_out = self.encoder(lt_embedded)
        combined = encoder_out + lt_embedded

        user_embedded = user_embedded[user_x]

        if self.args.at_type != 'none':
            at_embedded = self.at_net(timeslot_embedded, batch_data)
            combined = torch.cat([combined, at_embedded], dim=-1)

        user_embedded = user_embedded.unsqueeze(1).repeat(1, sequence_length, 1)
        combined = torch.cat([combined, user_embedded], dim=-1)

        if self.topic_num > 0:
            pre_embedded = self.user_net(pre_embedded).unsqueeze(1).repeat(1, sequence_length, 1)
            combined = torch.cat([combined, pre_embedded], dim=-1)

        out = self.fc_layer(combined.view(batch_size * sequence_length, combined.shape[2]))
        out = out.view(batch_size, sequence_length, -1)
        
        return out
        
    def calculate_loss(self, batch_data):
        loc_x = batch_data['POI_id']
        out = self.forward(batch_data)
        y_poi_id, seq_len = batch_data['y_POI_id']['POI_id'], batch_data['mask']
        B, L = batch_data['POI_id'].size()
        y_poi_seq = torch.full((B, L), 0, dtype=torch.long, device=self.device)
        
        for i in range(B):
            end = seq_len[i].item()
            y_poi_seq[i, :end] = torch.cat((loc_x[i, 1:end], y_poi_id[i].unsqueeze(dim=-1)), dim=-1)

        loss = self.loss_func(out.transpose(1, 2), y_poi_seq)
        
        return loss
    
    def predict(self, batch_data):
        out = self.forward(batch_data)

        batch_indices = torch.arange(batch_data['mask'].shape[0])
        y_pred = out[batch_indices, batch_data['mask'] - 1] 
        
        return y_pred

class ArrivalTime(nn.Module):
    def __init__(self, args):
        super(ArrivalTime, self).__init__()
        self.args = args
        self.base_dim = args.base_dim
        self.base_dim = args.base_dim
        self.num_heads = 4
        self.head_dim = self.base_dim // self.num_heads
        self.num_users = args.num_users
        self.base_dim = args.base_dim
        self.timeslot_num = 24

        if args.at_type == 'attn':
            self.user_preference = nn.Embedding(self.num_users, self.base_dim)
            self.w_q = nn.ModuleList(
                [nn.Linear(self.base_dim + self.base_dim, self.head_dim) for _ in range(self.num_heads)])
            self.w_k = nn.ModuleList(
                [nn.Linear(self.base_dim, self.head_dim) for _ in range(self.num_heads)])
            self.w_v = nn.ModuleList(
                [nn.Linear(self.base_dim, self.head_dim) for _ in range(self.num_heads)])
            self.unify_heads = nn.Linear(self.base_dim, self.base_dim)

    def forward(self, timeslot_embedded, batch_data):
        user_x = batch_data['user_id']
        hour_x = batch_data['hour']
        batch_size, sequence_length = hour_x.shape
        auxiliary_y = batch_data['timeslot_y']
        hour_mask = batch_data['hour_mask'].view(batch_size * sequence_length, -1)
        if self.args.at_type == 'truth':
            at_embedded = timeslot_embedded[auxiliary_y]
            return at_embedded
        if self.args.at_type == 'attn':
            hour_x = hour_x.view(batch_size * sequence_length)
            head_outputs = []
            user_preference = self.user_preference(user_x).unsqueeze(1).repeat(1, sequence_length, 1)
            user_feature = user_preference.view(batch_size * sequence_length, -1)
            time_feature = timeslot_embedded[hour_x]
            query = torch.cat([user_feature, time_feature], dim=-1)
            key = timeslot_embedded
            for i in range(self.num_heads):
                query_i = self.w_q[i](query)
                key_i = self.w_k[i](key)
                value_i = self.w_v[i](key)
                attn_scores_i = torch.matmul(query_i, key_i.T)
                scale = 1.0 / (key_i.size(-1) ** 0.5)
                attn_scores_i = attn_scores_i * scale
                attn_scores_i = attn_scores_i.masked_fill(hour_mask == 1, float('-inf'))
                attn_scores_i = torch.softmax(attn_scores_i, dim=-1)
                weighted_values_i = torch.matmul(attn_scores_i, value_i)
                head_outputs.append(weighted_values_i)
            head_outputs = torch.cat(head_outputs, dim=-1)
            head_outputs = head_outputs.view(batch_size, sequence_length, -1)
            return self.unify_heads(head_outputs)

        if self.args.at_type == 'static':
            time_trans_prob_mat = batch_data['prob_matrix_time_individual']
            at_embedded_user = torch.matmul(time_trans_prob_mat, timeslot_embedded)
            batch_indices = torch.arange(batch_size).view(-1, 1).expand_as(hour_x)
            at_embedded = at_embedded_user[batch_indices, hour_x, :]
            return at_embedded
