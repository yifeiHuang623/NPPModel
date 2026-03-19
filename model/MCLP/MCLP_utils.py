import torch.nn as nn
from torch.nn import init
import torch
import math

class PositionalEncoding(nn.Module):
    def __init__(self, emb_dim, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.emb_dim = emb_dim
        pos_encoding = torch.zeros(max_len, emb_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * -(math.log(10000.0) / emb_dim))
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        pos_encoding = pos_encoding.unsqueeze(0)
        self.register_buffer("pos_encoding", pos_encoding)
        self.dropout = nn.Dropout(0.1)

    def forward(self, out):
        out = out + self.pos_encoding[:, :out.size(1)].detach()
        out = self.dropout(out)
        return out


class MyEmbedding(nn.Module):
    def __init__(self, args):
        super(MyEmbedding, self).__init__()
        self.args = args

        self.num_locations = args.num_pois
        self.base_dim = args.base_dim
        self.num_users = args.num_users

        self.user_embedding = nn.Embedding(self.num_users, self.base_dim)
        self.location_embedding = nn.Embedding(self.num_locations, self.base_dim)
        self.timeslot_embedding = nn.Embedding(24, self.base_dim)

    def forward(self, batch_data):
        location_x = batch_data['POI_id']

        loc_embedded = self.location_embedding(location_x)
        user_embedded = self.user_embedding(torch.arange(end=self.num_users, dtype=torch.int, device=location_x.device))

        timeslot_embedded = self.timeslot_embedding(torch.arange(end=24, dtype=torch.int, device=location_x.device))

        return loc_embedded, timeslot_embedded, user_embedded

class UserNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(UserNet, self).__init__()
        self.topic_num = input_dim
        self.output_dim = output_dim

        self.block = nn.Sequential(
            nn.Linear(self.topic_num, self.topic_num * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.topic_num * 2, self.topic_num),
        )
        self.final = nn.Sequential(
            nn.LayerNorm(self.topic_num),
            nn.Linear(self.topic_num, self.output_dim)
        )

    def forward(self, topic_vec):
        x = topic_vec
        topic_vec = self.block(topic_vec)
        topic_vec = x + topic_vec

        return self.final(topic_vec)
    
class MyFullyConnect(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MyFullyConnect, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(input_dim, input_dim*2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim*2, input_dim),
            nn.Dropout(0.1),
        )

        self.batch_norm = nn.BatchNorm1d(input_dim)
        self.drop = nn.Dropout(0.1)

        num_locations = output_dim
        self.linear_class1 = nn.Linear(input_dim, num_locations)

    def forward(self, out):
        x = out
        out = self.block(out)
        out = out + x
        out = self.batch_norm(out)
        out = self.drop(out)

        return self.linear_class1(out)
    
class TransEncoder(nn.Module):
    def __init__(self, args):
        super(TransEncoder, self).__init__()
        self.args = args
        input_dim = args.base_dim

        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim,
                                                   activation='gelu',
                                                   batch_first=True,
                                                   dim_feedforward=input_dim,
                                                   nhead=4,
                                                   dropout=0.1)

        encoder_norm = nn.LayerNorm(input_dim)

        # Transformer Encoder
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer,
                                             num_layers=2,
                                             norm=encoder_norm)
        self.initialize_parameters()

    def forward(self, embedded_out, src_mask):
        out = self.encoder(embedded_out, mask=src_mask)

        return out

    def initialize_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                init.xavier_uniform_(p)


class LSTMEncoder(nn.Module):
    def __init__(self, args):
        super(LSTMEncoder, self).__init__()
        input_dim = args.Embedding.base_dim

        self.encoder = nn.LSTM(input_size=input_dim,
                               hidden_size=input_dim,
                               num_layers=2,
                               dropout=0.1,
                               batch_first=True)
        self.initialize_parameters()

    def forward(self, out):
        out, _ = self.encoder(out)

        return out

    def initialize_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                init.xavier_uniform_(p)
